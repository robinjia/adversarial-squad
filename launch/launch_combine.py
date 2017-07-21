"""Insert a description of this module."""
import argparse
import subprocess
import sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('Insert a description of this script.')
  parser.add_argument('description')
  parser.add_argument('bundles', nargs='+', help='Codalab bundles')
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()

def main():
  deps = [':combine_data.py'] + [':%s' % b for b in OPTS.bundles]
  cmd_str = 'python combine_data.py all_data.json %s' % (' '.join(OPTS.bundles))
  args = ['cl', 'run'] + deps + [cmd_str, '--request-cpus', '4', 
                                 '-n', 'combine', '-d', OPTS.description]
  subprocess.call(args)


if __name__ == '__main__':
  OPTS = parse_args()
  main()

