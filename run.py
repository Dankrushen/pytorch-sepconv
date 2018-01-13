#!/usr/bin/env python2.7

import sys
import getopt
import math
import numpy
import torch
import torch.utils.serialization
import PIL
import PIL.Image
from decimal import Decimal
from datetime import datetime
import time
import os
import shutil

# If the user has not installed ffmpeg
import imageio
imageio.plugins.ffmpeg.download()

from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from SeparableConvolution import SeparableConvolution # the custom SeparableConvolution layer

torch.cuda.device(1) # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'lf'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './result.png'
arguments_strVideo = ''
arguments_strVideoOut = ''
arguments_strVideoAudio = ''

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model':
		arguments_strModel = strArgument # which model to use, l1 or lf, please see our paper for more details

	elif strOption == '--first':
		arguments_strFirst = strArgument # path to the first frame

	elif strOption == '--second':
		arguments_strSecond = strArgument # path to the second frame

	elif strOption == '--out':
		arguments_strOut = strArgument # path to where the output should be stored

	elif strOption == '--video':
		arguments_strVideo = strArgument # path to the input video

	elif strOption == '--video-out':
		arguments_strVideoOut = strArgument # path to the output video
	elif strOption == '--video-audio':
		arguments_strVideoAudio = strArgument # path to the video's audio
	# end
# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		def Basic(intInput, intOutput):
			return torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False)
			)
		# end

		def Subnet():
			return torch.nn.Sequential(
				torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Conv2d(in_channels=64, out_channels=51, kernel_size=3, stride=1, padding=1),
				torch.nn.ReLU(inplace=False),
				torch.nn.Upsample(scale_factor=2, mode='bilinear'),
				torch.nn.Conv2d(in_channels=51, out_channels=51, kernel_size=3, stride=1, padding=1)
			)
		# end

		self.moduleConv1 = Basic(6, 32)
		self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

		self.moduleConv2 = Basic(32, 64)
		self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

		self.moduleConv3 = Basic(64, 128)
		self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

		self.moduleConv4 = Basic(128, 256)
		self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

		self.moduleConv5 = Basic(256, 512)
		self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

		self.moduleDeconv5 = Basic(512, 512)
		self.moduleUpsample5 = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear'),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleDeconv4 = Basic(512, 256)
		self.moduleUpsample4 = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear'),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleDeconv3 = Basic(256, 128)
		self.moduleUpsample3 = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear'),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleDeconv2 = Basic(128, 64)
		self.moduleUpsample2 = torch.nn.Sequential(
			torch.nn.Upsample(scale_factor=2, mode='bilinear'),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleVertical1 = Subnet()
		self.moduleVertical2 = Subnet()
		self.moduleHorizontal1 = Subnet()
		self.moduleHorizontal2 = Subnet()

		self.modulePad = torch.nn.ReplicationPad2d([ int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)), int(math.floor(51 / 2.0)) ])

		self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
	# end

	def forward(self, variableInput1, variableInput2):
		variableJoin = torch.cat([variableInput1, variableInput2], 1)

		variableConv1 = self.moduleConv1(variableJoin)
		variablePool1 = self.modulePool1(variableConv1)

		variableConv2 = self.moduleConv2(variablePool1)
		variablePool2 = self.modulePool2(variableConv2)

		variableConv3 = self.moduleConv3(variablePool2)
		variablePool3 = self.modulePool3(variableConv3)

		variableConv4 = self.moduleConv4(variablePool3)
		variablePool4 = self.modulePool4(variableConv4)

		variableConv5 = self.moduleConv5(variablePool4)
		variablePool5 = self.modulePool5(variableConv5)

		variableDeconv5 = self.moduleDeconv5(variablePool5)
		variableUpsample5 = self.moduleUpsample5(variableDeconv5)

		variableCombine = variableUpsample5 + variableConv5

		variableDeconv4 = self.moduleDeconv4(variableCombine)
		variableUpsample4 = self.moduleUpsample4(variableDeconv4)

		variableCombine = variableUpsample4 + variableConv4

		variableDeconv3 = self.moduleDeconv3(variableCombine)
		variableUpsample3 = self.moduleUpsample3(variableDeconv3)

		variableCombine = variableUpsample3 + variableConv3

		variableDeconv2 = self.moduleDeconv2(variableCombine)
		variableUpsample2 = self.moduleUpsample2(variableDeconv2)

		variableCombine = variableUpsample2 + variableConv2

		variableDot1 = SeparableConvolution()(self.modulePad(variableInput1), self.moduleVertical1(variableCombine), self.moduleHorizontal1(variableCombine))
		variableDot2 = SeparableConvolution()(self.modulePad(variableInput2), self.moduleVertical2(variableCombine), self.moduleHorizontal2(variableCombine))
		
		return variableDot1 + variableDot2
	# end
# end

moduleNetwork = Network().cuda()

##########################################################

def process(tensorInputFirst, tensorInputSecond, tensorOutput):
	assert(tensorInputFirst.size(1) == tensorInputSecond.size(1))
	assert(tensorInputFirst.size(2) == tensorInputSecond.size(2))

	intWidth = tensorInputFirst.size(2)
	intHeight = tensorInputFirst.size(1)

	# assert(intWidth <= 1280) # while our approach works with larger images, we do not recommend it unless you are aware of the implications
	# assert(intHeight <= 720) # while our approach works with larger images, we do not recommend it unless you are aware of the implications

	intPaddingLeft = int(math.floor(51 / 2.0))
	intPaddingTop = int(math.floor(51 / 2.0))
	intPaddingRight = int(math.floor(51 / 2.0))
	intPaddingBottom = int(math.floor(51 / 2.0))
	modulePaddingInput = torch.nn.Sequential()
	modulePaddingOutput = torch.nn.Sequential()

	if True:
		intPaddingWidth = intPaddingLeft + intWidth + intPaddingRight
		intPaddingHeight = intPaddingTop + intHeight + intPaddingBottom

		if intPaddingWidth != ((intPaddingWidth >> 7) << 7):
			intPaddingWidth = (((intPaddingWidth >> 7) + 1) << 7) # more than necessary
		# end
		
		if intPaddingHeight != ((intPaddingHeight >> 7) << 7):
			intPaddingHeight = (((intPaddingHeight >> 7) + 1) << 7) # more than necessary
		# end

		intPaddingWidth = intPaddingWidth - (intPaddingLeft + intWidth + intPaddingRight)
		intPaddingHeight = intPaddingHeight - (intPaddingTop + intHeight + intPaddingBottom)

		modulePaddingInput = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight + intPaddingWidth, intPaddingTop, intPaddingBottom + intPaddingHeight])
		modulePaddingOutput = torch.nn.ReplicationPad2d([0 - intPaddingLeft, 0 - intPaddingRight - intPaddingWidth, 0 - intPaddingTop, 0 - intPaddingBottom - intPaddingHeight])
	# end

	if True:
		tensorInputFirst = tensorInputFirst.cuda()
		tensorInputSecond = tensorInputSecond.cuda()

		modulePaddingInput = modulePaddingInput.cuda()
		modulePaddingOutput = modulePaddingOutput.cuda()
	# end

	if True:
		variablePaddingFirst = modulePaddingInput(torch.autograd.Variable(data=tensorInputFirst.view(1, 3, intHeight, intWidth), volatile=True))
		variablePaddingSecond = modulePaddingInput(torch.autograd.Variable(data=tensorInputSecond.view(1, 3, intHeight, intWidth), volatile=True))
		variablePaddingOutput = modulePaddingOutput(moduleNetwork(variablePaddingFirst, variablePaddingSecond))

		tensorOutput.resize_(3, intHeight, intWidth).copy_(variablePaddingOutput.data[0])
	# end

	if True:
		tensorInputFirst.cpu()
		tensorInputSecond.cpu()
		tensorOutput.cpu()
	# end
#end

tensorOutput = torch.FloatTensor()

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 2, length = 50, fill = '#', emptyFill = '='):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + emptyFill * (length - filledLength)
	sys.stdout.write('\r%s[%s] %s%%%s' % (prefix, bar, percent, suffix))
	sys.stdout.flush()
	# Print New Line on Complete
	if iteration == total: 
		print("")

if arguments_strVideo and arguments_strVideoOut:
	# Process video
	exists = os.path.exists(arguments_strVideoOut) and os.path.isfile(arguments_strVideoOut)
	startFrame = 0

	if exists:
		# Continue from where it left off
		print("Resuming processing...")
		
		print("Moving processed video to temporary storage...")
		tempDir = "temp"
		tempDest = tempDir + "/" + os.path.basename(arguments_strVideoOut)
		if os.path.exists(tempDir):
			shutil.rmtree(tempDir)
		os.makedirs(tempDir)
		
		shutil.move(arguments_strVideoOut, tempDest)
		
		readerCont = FFMPEG_VideoReader(tempDest, False)
		totalFramesCont = readerCont.nframes
		if totalFramesCont % 2 == 0:
			# Even numbers: Ended on interpolated frame
			# startFrame = number of last original frame
			# This means that it will skip this frame
			startFrame = totalFramesCont / 2
		else:
			# Odd numbers: Ended on original frame
			# startFrame = number of second last original frame
			# This means that it will move on to the current frame
			startFrame = (totalFramesCont - 1) / 2

	reader = FFMPEG_VideoReader(arguments_strVideo, False)
	if arguments_strVideoAudio:
		writer = FFMPEG_VideoWriter(arguments_strVideoOut, reader.size, reader.fps*2, audiofile=arguments_strVideoAudio)

	else: 
		writer = FFMPEG_VideoWriter(arguments_strVideoOut, reader.size, reader.fps*2)

	if exists:
		# Write frames that were already completed
		print("Re-writing processed frames...")
		printProgressBar(0, readerCont.nframes)
		for x in range(0, readerCont.nframes):
			writer.write_frame(readerCont.read_frame())
			printProgressBar(x + 1, readerCont.nframes)

		reader.skip_frames(startFrame)
		
		readerCont.close()
		
		print("Deleting temporary file(s)...")
		shutil.rmtree(tempDir)
		
		print("Processing resumed!")
		print("")
		
	totalFrames = reader.nframes
	nextFrame = reader.read_frame()
	startedTime = datetime.now()
	for x in range(startFrame, reader.nframes):
		start = datetime.now()
		firstFrame = nextFrame
		nextFrame = reader.read_frame()
		tensorInputFirst = torch.FloatTensor(numpy.rollaxis(firstFrame[:,:,::-1], 2, 0) / 255.0)
		tensorInputSecond = torch.FloatTensor(numpy.rollaxis(nextFrame[:,:,::-1], 2, 0) / 255.0)
		process(tensorInputFirst, tensorInputSecond, tensorOutput)
		writer.write_frame(firstFrame)
		writer.write_frame((numpy.rollaxis(tensorOutput.clamp(0.0, 1.0).numpy(), 0, 3)[:,:,::-1] * 255.0).astype(numpy.uint8))
		
		# Calculate and display stats
		countDone = x + 1
		percentDone = round(Decimal((float(countDone) / float(totalFrames)) * 100), 2)
		print("Frames done: " + str(countDone) + "/" + str(totalFrames) + " (" + str(percentDone) + "%)")
		secondsDone = round(Decimal((float(countDone) / float(reader.fps))), 2)
		print("Seconds of video done: " + str(secondsDone))
		timeTaken = datetime.now() - start
		fps = round(Decimal(float(float(1)/timeTaken.total_seconds())), 3)
		print("Frames per second: " + str(fps) + " f/s")
		spf = round(Decimal(timeTaken.total_seconds()), 3)
		print("Seconds per frame: " + str(spf) + " s/f")
		timeElapsed = datetime.now() - startedTime
		m, s = divmod(timeElapsed.total_seconds(), 60)
		h, m = divmod(m, 60)
		timeElapsedSeconds = "%dH:%02dM:%02dS" % (h, m, s)
		print("Elapsed time: " + timeElapsedSeconds)
		timeLeft = float(timeTaken.total_seconds() * float(totalFrames - countDone))
		m, s = divmod(timeLeft, 60)
		h, m = divmod(m, 60)
		timeLeftSeconds = "%dH:%02dM:%02dS" % (h, m, s)
		print("Estimated time remaining: " + timeLeftSeconds)
		print("")
	#end
	writer.write_frame(nextFrame)
	writer.close()
	reader.close()
	print("Done!")
else:
	# Process image
	tensorInputFirst = torch.FloatTensor(numpy.rollaxis(numpy.asarray(PIL.Image.open(arguments_strFirst))[:,:,::-1], 2, 0).astype(numpy.float32) / 255.0)
	tensorInputSecond = torch.FloatTensor(numpy.rollaxis(numpy.asarray(PIL.Image.open(arguments_strSecond))[:,:,::-1], 2, 0).astype(numpy.float32) / 255.0)	
	process(tensorInputFirst, tensorInputSecond, tensorOutput)
	PIL.Image.fromarray((numpy.rollaxis(tensorOutput.clamp(0.0, 1.0).numpy(), 0, 3)[:,:,::-1] * 255.0).astype(numpy.uint8)).save(arguments_strOut)
#end
