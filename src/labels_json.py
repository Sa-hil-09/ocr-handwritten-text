import json
import random
import cv2
import numpy as np
import os

"""
{
    "img1.jpg": "abc xyz",
    ...
    "imgn.jpg": "def ghi"
}
"""
class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath):
		self.filePath = filePath
		self.gtText = gtText



class DataLoader:
	"loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database" 

	def __init__(self, filePath, maxTextLen):
		"loader for dataset at given location, preprocess images and text according to parameters"

		assert filePath[-1]=='/'
		self.samples = []
	
		f=open(filePath+'lines.txt')
		chars = set()
		bad_samples = []
        #bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
		for line in f:
			# ignore comment line
			if not line or line[0]=='#':
				continue
			
			lineSplit = line.strip().split(' ')
			assert len(lineSplit) >= 9
			
			# filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
			fileNameSplit = lineSplit[0].split('-')
			#fileName = filePath + 'lines/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'
			fileName = filePath + 'lines/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

            # GT text are columns starting at 9
			gtText = self.truncateLabel(' '.join(lineSplit[8].split('|')), maxTextLen)
			chars = chars.union(set(list(gtText)))           

			# check if image is not empty
			if not os.path.getsize(fileName):
				bad_samples.append(lineSplit[0] + '.png')
				continue
			#print(type(gtText), type(fileName))
			# put sample into dict
			#self.samples[fileName]= self.samples[gtText]

			# put sample into list
			self.samples.append(Sample(gtText, fileName))


	def truncateLabel(self, text, maxTextLen):
		# ctc_loss can't compute loss if it cannot find a mapping between text label and input 
		# labels. Repeat letters cost double because of the blank symbol needing to be inserted.
		# If a too-long label is provided, ctc_loss returns an infinite gradient
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text
    

if __name__ == "__main__":
    loader = DataLoader('../data/', 100)
    #y = json.dumps(loader.samples)
    samples={}
    #print(loader.samples)
    for obj in loader.samples:
        print(obj.filePath,obj.gtText)
        samples[obj.filePath]= obj.gtText
    #print(samples)
    y = json.dumps(samples)
    with open('data.json', 'w', encoding='utf-8') as outfile:
        json.dump(samples, outfile, ensure_ascii=False, indent=2)