from glob import glob as glob
import multiprocessing
import functools
import sys
import os

pn_CPP_FILES = os.path.join("data", "imgproc", "cpp")
fn_CPP_OUT = os.path.join("data", "imgproc", "imgproc.out")

class Image():
	def __init__(self, src, out):
		self.src = src
		self.out = os.path.join(out, src.split('/')[-1])


def compile():
	cpp = " ".join(sorted(glob(os.path.join(pn_CPP_FILES, "*.cpp"))))
	cmd = "g++ %s -o %s" % (cpp, fn_CPP_OUT)

	if sys.platform == "linux":
		cmd += " -std=c++17 -lstdc++fs `pkg-config --cflags --libs opencv`"

	if os.system(cmd) != 0:
		print("\nError with `opencv` tag.\nCompiling with `opencv4`...\n")

		if os.system(cmd.replace("opencv", "opencv4")) != 0:
			print("Preprocess compilation error.")
		else:
			print("Preprocess compiled successfully.")
	else:
		print("Preprocess compiled successfully.")


def execute(images, out):
	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	pool.map(functools.partial(__execute__, out=out), images)
	pool.close()
	pool.join()


def __execute__(image, out):
	im = Image(image, out)
	cmd = " ".join([fn_CPP_OUT, im.src, im.out])

	if os.system(cmd) != 0:
		sys.exit("Image process error: %s" % im.src)

if not os.path.isfile(fn_CPP_OUT):
	compile()