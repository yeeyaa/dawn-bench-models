import argparse
import os
import subprocess
import sys

def main(checkpoint_path, model, use_bottleneck):
  print("Run Using Checkpoint\tNumber of images\tInference time (milliseconds)")
  num_trials = 10000
  cnt = 1
  while True:
    ckpt_path = ("%5d" % cnt).replace(' ', '0')
    full_ckpt_path = os.path.join(checkpoint_path, ckpt_path)
    if not os.path.exists(full_ckpt_path):
      break
    # SimonChange: For dawnbench inference time. Change batch_size to 1 and run for 10000 batches. (Total cifar10 test images)
    # SimonChnage: ref: https://github.com/stanford-futuredata/dawn-bench-entries
    #for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    for batch_size in [1]:
      command = ("python3 resnet/resnet_main.py --mode=eval --eval_data_path=cifar10/test_batch.bin "
                 "--eval_dir=data/%(model)s/log_root/eval --dataset='cifar10' --model=%(model)s "
                 "--use_bottleneck=%(use_bottleneck)s --eval_batch_count=%(num_trials)d --eval_once=True --num_gpus=1 "
                 "--data_format=NHWC --time_inference=True --batch_size=%(batch_size)d" %
                 {"model": model, "use_bottleneck": "True" if use_bottleneck else "False", "batch_size": batch_size,
                  "num_trials": num_trials})
      full_command = command + " --log_root=%s 2>/dev/null" % full_ckpt_path
      try:
        output = subprocess.check_output(full_command, shell=True)
        output = output.decode('utf8').strip()
        for line in output.split('\n'):
          if "Time for inference" in line:
            line = line.strip()
            inference_time = (float(line.split(": ")[1]) / num_trials) * 1000
            stats = [ckpt_path, batch_size, inference_time]
            print("\t".join([str(stat) for stat in stats]))
            sys.stdout.flush()
            cnt += 1
      except:
        stats = [batch_size, ""]
        print("\t".join([str(stat) for stat in stats]))
        sys.stdout.flush()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description=("Backup model checkpoints periodically")
  )
  parser.add_argument('-i', "--checkpoint_path", type=str, required=True,
                      help="Path to dumped model checkpoints")
  parser.add_argument('-m', "--model", type=str, required=True,
                      help="Model name")
  parser.add_argument('-b', "--use_bottleneck", type=bool, default=False,
                      help="Use bottleneck")

  cmdline_args = parser.parse_args()
  opt_dict = vars(cmdline_args)

  checkpoint_path = opt_dict["checkpoint_path"]
  model = opt_dict["model"]
  use_bottleneck = opt_dict["use_bottleneck"]

  main(checkpoint_path, model, use_bottleneck)
