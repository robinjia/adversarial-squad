import json
import os
import sys

def main():
  out_file = sys.argv[1]
  bundles = sys.argv[2:]
  all_data = []
  version = None
  for b in bundles:
    with open(os.path.join(b, 'adversarial_data.json')) as f:
      cur_data = json.load(f)
    all_data.extend(cur_data['data'])
    version = cur_data['version']
  out_obj = {'data': all_data, 'version': version}
  with open(out_file, 'w') as f:
    json.dump(out_obj, f)

if __name__ == '__main__':
  main()
