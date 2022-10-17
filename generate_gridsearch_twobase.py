import argparse
import os
import pathlib
from hposcripts.hposcripts import generator

def main():

    args = _get_args()
    hpo = generator()

    sbatch_opts = {
        'job-name': args.outputname,
        'time': args.stime,
        'partition': args.squeue,
        'chdir' : args.work_dir,
        'mem' : args.smem,
        'gpus' : 1,
        # 'exclude' : 'gpu004,gpu005,gpu006,gpu007',
    }

    hpo.set_sbatch_opts(sbatch_opts)
    
    ###options to test

    left_data         = ["'ring'",
                         "'concentricrings'",
                         "'fourcircles'",
                         "'checkerboard'",
                         "'spirals'"]
    right_data        = [
                        # "'ring'",
                        #  "'concentricrings'",
                        #  "'fourcircles'",
                         "'checkerboard'",
                        #  "'spirals'"
                         ]
    f4f_dir           = [
                         "'alternate'",
                         "'forward'", 
                         "'inverse'",
                        #  "'both'",
                        #  "'iterate'"
                        ]

    hpo.add_opt('base_dist.left.data',left_data,True)
    hpo.add_opt('base_dist.right.data',right_data,True)
    hpo.add_opt('top_transformer.direction',f4f_dir,True)

    hpo.add_script_line('export XDG_RUNTIME_DIR=""')
    hpo.add_script_line('module load GCC/9.3.0 Singularity/3.7.3-Go-1.14',lastline = True)


    runfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),'run_twobase_nocond.py')
    cmd = '\nsrun singularity exec --nv'
    if args.singularity_mounts is not None:
        cmd += f' -B {args.singularity_mounts}'
    cmd += f' {args.singularity_instance}\\\n\tpython3 {runfile} output.save_dir={args.outputdir} output.name={args.outputname}_${{SLURM_ARRAY_TASK_ID}}\\\n\t\t'

    pathlib.Path(args.work_dir).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.sbatch_output).parent.mkdir(parents=True, exist_ok=True)
    
    hpo.set_command(cmd)
    hpo.launch(outputfile=args.sbatch_output,submit=args.submit)

    return

def _get_args():
    parser = argparse.ArgumentParser()

    ##General settings
    parser.add_argument('-d', '--outputdir', type=str,
                        help='Choose the base output directory',
                        required=True)

    parser.add_argument('-n', '--outputname', type=str,
                        help='Set the output name directory',
                        required=True)
    
    parser.add_argument('--squeue',type=str,default='private-dpnc-gpu,shared-gpu')
    parser.add_argument('--stime',type=str,default='00-12:00:00')
    parser.add_argument('--smem',type=str,default='25GB')
    parser.add_argument('--work-dir',type=str,required=True)

    parser.add_argument('--submit',action='store_true',
                        dest='submit')

    parser.add_argument('--sbatch-output',type=str,default='gridsearch_submit.txt')

    parser.add_argument('--singularity-instance',type=str,default='/srv/beegfs/scratch/groups/rodem/flows4flows/curtains-flows4flows.simg')
    parser.add_argument('--singularity-mounts',type=str,default='/home,/srv')

    parser.set_defaults(submit=False)
    return parser.parse_args()

if __name__ =='__main__':
    main()