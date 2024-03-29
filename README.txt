run ./test.py to test the HVS-JND
First, you have to change line 157:
	images, raw_images = load_images(str("/mnt/disk10T/Henry/src_imgs/") + str(number) + str(".png"))
	this path is the test data
Then, run python test.py
