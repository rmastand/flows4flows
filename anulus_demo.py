from pathlib import Path

from matplotlib import pyplot as plt
from nflows import transforms
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.utils import tensor2numpy
from torch.utils.data import DataLoader

from ffflows.data.plane import Anulus, ConditionalAnulus
from ffflows.models import FlowForFlow, DeltaFlowForFlow
from torch.nn import functional as F
import torch


def spline_inn(inp_dim, nodes=128, num_blocks=2, num_stack=3, tail_bound=3.5, tails='linear', activation=F.relu, lu=0,
               num_bins=10, context_features=None):
    transform_list = []
    for i in range(num_stack):
        transform_list += [
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, nodes,
                                                                               num_blocks=num_blocks,
                                                                               tail_bound=tail_bound,
                                                                               num_bins=num_bins,
                                                                               tails=tails, activation=activation,
                                                                               context_features=context_features)]
        if lu:
            transform_list += [transforms.LULinear(inp_dim)]
        else:
            transform_list += [transforms.ReversePermutation(inp_dim)]

    return transforms.CompositeTransform(transform_list[:-1])


def plot_training(training, validation):
    fig, ax = plt.subplots(1, 1)
    ax.plot(tensor2numpy(training), label='Training')
    ax.plot(tensor2numpy(validation), label='Validation')
    ax.legend()
    return fig


def train(model, train_loader, valid_loader, n_epochs, learning_rate, device, directory, sv_nm=None):
    directory.mkdir(exist_ok=True, parents=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_steps = len(train_loader) * n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, last_epoch=-1,
                                                           eta_min=0)
    train_loss = torch.zeros(n_epochs)
    valid_loss = torch.zeros(n_epochs)
    for epoch in range(n_epochs):
        print(f"Training epoch {epoch}")
        t_loss = []
        # Redefine the loaders at the start of every epoch, this will also resample if resampling the mass
        for step, data in enumerate(train_loader):
            model.train()
            # Zero the accumulated gradients
            optimizer.zero_grad()
            # Get the loss
            loss = model.compute_loss(data)
            # Calculate the derivatives
            loss.backward()
            # Step the optimizers and schedulers
            optimizer.step()
            scheduler.step()
            # Store the loss
            t_loss += [loss.item()]
        # Save the model
        torch.save(model.state_dict(), directory / f'{epoch}')

        # Store the losses across the epoch and start the validation
        train_loss[epoch] = torch.tensor(t_loss).mean()
        v_loss = torch.zeros(len(valid_loader))
        for v_step, (data) in enumerate(valid_loader):
            with torch.no_grad():
                v_loss[v_step] = model.compute_loss(data)
        valid_loss[epoch] = v_loss.mean()

    if sv_nm is not None:
        # Training and validation losses
        fig = plot_training(train_loss, valid_loss)
        fig.savefig(sv_nm)
        # fig.show()
        plt.close(fig)

    model.eval()
    return train_loss, valid_loss


def plot_data(data, nm):
    data = tensor2numpy(data)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bound = 4
    bounds = [[-bound, bound], [-bound, bound]]
    ax.hist2d(data[:, 0], data[:, 1], bins=256)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    plt.savefig(nm)


def shift_anulus():
    # Hyperparameters
    batch_size = 128

    # Experiment path
    top_save = Path('results/anulus_demo4')
    top_save.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define data and loader
    def get_loader(n_points=int(1e4), batch_size=128):
        dataset = ConditionalAnulus(num_points=n_points)
        return DataLoader(dataset=dataset, batch_size=batch_size)

    train_loader = get_loader(batch_size=batch_size)
    valid_loader = get_loader(n_points=int(1e4), batch_size=1000)

    class ConditionalFlow(Flow):
        """For the demo the way the conditioning is done is hard coded."""

        def split_data(self, data):
            return data[0], data[1]

        def compute_loss(self, data):
            data, context = data[0], data[1]
            return -self.log_prob(data, context=context).mean()

    # Define the base density
    base_density = ConditionalFlow(spline_inn(2, context_features=1), StandardNormal([2]))

    # # Define the flow for flow object
    # flow_for_flow = FlowForFlow(spline_inn(2, context_features=1), base_density)

    class fff(DeltaFlowForFlow):
        """Try and test with current setup"""

        def split_data(self, data):
            return data[0],data[1]

        def compute_loss(self, data):
            data, context = data[0],data[1]
            # Map to a random
            return -self.log_prob(data, context, context + 0.3 * torch.randn(len(context)).view(-1, 1)).mean()

        def final_eval(self, data, context_l, context_r):
            # data, context = data[0], data[1]
            context_r = context_l + context_r
            c_l = torch.ones(len(data))*context_l if len(data) != 0 else torch.ones(len(data)) 
            c_r = torch.ones(len(data))*context_r if len(data) != 0 else torch.ones(len(data))
            return self.transform(data, c_l, c_r)

    flow_for_flow = fff(spline_inn(2, context_features=1), base_density)

    # Train the base density
    models_save = top_save / 'base_densities'
    models_save.mkdir(exist_ok=True)
    train(base_density, train_loader, valid_loader, 1, 0.001, device, models_save, top_save / 'loss_base.png')

    # Evaluate the base density
    for rad in [1, 2, 3]:
        with torch.no_grad():
            samples = base_density.sample(1, context=rad * torch.ones((int(1e5), 1))).squeeze(1)
        plot_data(samples, top_save / f'base_density_{rad}.png')

    # Train the flow for flow model
    # Need to be able to turn off gradients for the base density
    def no_more_grads(model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    no_more_grads(base_density)

    models_save = top_save / 'f4fs'
    models_save.mkdir(exist_ok=True)
    train(flow_for_flow, train_loader, valid_loader, 1, 0.001, device, models_save, top_save / 'loss_f4f.png')

    n_points = int(1000)
    input_dist = ConditionalAnulus(num_points=n_points, radius=0.5)
    nocond = torch.stack([d[0] for d in input_dist.data])
    plot_data(nocond, top_save / f'flow_for_flow_input.png')

    # for rad in [-1.5, -1, -0.5, 1, 1.5, 2]:
    for inner_rad in [0.3, 0.5, 0.7]:
        input_dist = ConditionalAnulus(num_points=n_points, radius=inner_rad)
        nocond = torch.stack([d[0] for d in input_dist.data])
        plot_data(nocond, top_save / f'flow_for_flow_input_{inner_rad}.png')

        for shift in [0.1, 0.2, 0.3]:
            with torch.no_grad():
                samples, _ = flow_for_flow.final_eval(nocond, inner_rad, shift)
            plot_data(samples, top_save / f'flow_for_flow_output_{inner_rad}_plus_{shift}.png')


if __name__ == '__main__':
    shift_anulus()
