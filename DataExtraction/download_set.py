from quickdraw import QuickDrawData
from quickdraw import QuickDrawDataGroup
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true", default="True")
parser.add_argument("-o", "--output", help="where to write data set", 
					 default= "./out")
parser.add_argument("-c", "--count", help="how many images to generate from each type",
					type=int, default="1")

args = parser.parse_args()


try:
    os.mkdir(args.output)
except OSError:
  	 print ("Directory already exist:  %s" % args.output)	
else:
	if args.verbose:
  	 print ("Successfully created the directory %s" % args.output)

print(args)
qd = QuickDrawData(recognized=None, max_drawings=1000, refresh_data=False, jit_loading=True, print_messages=False, cache_dir='./.quickdrawcache')

print(f'Total categories count {len(qd.drawing_names)}')
currentIteration = 0
for name in qd.drawing_names:

	group =  QuickDrawDataGroup(name)
	for i in range(0, args.count):
		data_point = group.get_drawing()
		img = data_point.image.resize((32,32))
		img.save(f'{args.output}/{name}_{data_point.key_id}.png')
	if args.verbose:
		currentIteration += 1
		os.system("cls")
		print(f'Images loaded [{name}]: {currentIteration}/{len(qd.drawing_names)}')
