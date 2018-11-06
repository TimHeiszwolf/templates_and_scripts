import os
from os import listdir
from os.path import isfile, join
import shutil

def getDir(specifications):
	"""
	Gets a directory (with specifications of the user)
	"""
	while True:
		print('')
		print(specifications)
		print('The input needs to be given relative to the directory which this script is in.')
		print('The current directory is "' + os.getcwd() + '\\".')
		directory = '.\\' + input('Name directory: ')
		print('Directory is: "' + directory + '".')
		
		if os.path.isdir(directory):
			break
		else:
			print('Not a directory.')
		
	return directory
		
def getAllFilesInDir(directory):
	"""
	Gets all files in the directory
	"""
	files = []
	
	for root, dirs, file in os.walk(directory):	
		files = files + [join(root, name) for name in file]
	
	return(files)

def ReplaceAllDocs(replacements, docs):
	"""
	Replaces all files with equal filenames.
	"""
	for replacement in replacements:
		split = replacement.split("\\")
		repname = split[len(split)-1]
		replaced = False
		
		for doc in docs:
			split = doc.split("\\")
			docname = split[len(split)-1]
			
			if repname==docname:
				shutil.copy2(replacement, doc)
				replaced = True
		
		if (not replaced):
			print("Didn't replace any file with " + replaced.name)

while True:
	replacements = getAllFilesInDir(getDir("What is the directory of the replacement files?"))
	docs = getAllFilesInDir(getDir("What is the directory of the files to be replaced?"))
	
	if replacements != docs:
		break
	else:
		print("")
		print("Same directory")
		print("")

ReplaceAllDocs(replacements, docs)
