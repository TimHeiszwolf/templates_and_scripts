#This script asks for a directory and then adds a pre or suffix to each filename in that directory.

import os

def list_files(startpath):
	"""https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python"""
	for root, dirs, files in os.walk(startpath):
		level = root.replace(startpath, '').count(os.sep)
		indent = ' ' * 4 * (level)
		print('{}{}/'.format(indent, os.path.basename(root)))
		subindent = ' ' * 4 * (level + 1)
		for f in files:
			print('{}{}'.format(subindent, f))

list_files('.')

while True:
	print('')
	print('The target needs to be a directory and the input needs to be given relative to the directory which this script is in.')
	print('The current directory is "'+os.getcwd()+'\\".')
	target='.\\'+input('Name target directory: ')
	print('Target is: "'+target+'".')
	
	if os.path.isdir(target):
		break
	else:
		print('Not a directory.')

while True:
	print('')
	PreOrSuf=input('Do you want a prefix or a suffix? Anwser "pre" or "suf":').lower()
	
	if (PreOrSuf=='pre' or PreOrSuf=='suf') and (input('Anwser is "'+PreOrSuf+'". Correct? (Y/N): ')=='Y'):
		break

while True:
	print('')
	text=input('What should the '+PreOrSuf+'fix be?: ')
	
	if input('The '+PreOrSuf+'fix is "'+text+'". Correct? (Y/N): ')=='Y':
		break

for filename in os.listdir(target):
	#https://stackoverflow.com/questions/2759067/rename-multiple-files-in-a-directory-in-python
	#https://stackoverflow.com/questions/225735/batch-renaming-of-files-in-a-directory
	if PreOrSuf=='pre':
		os.rename(os.path.join(target, filename), os.path.join(target,text+filename))
	
	if PreOrSuf=='suf':
		Ext=os.path.splitext(os.path.join(target, filename))[1]#https://stackoverflow.com/questions/541390/extracting-extension-from-filename-in-python
		os.rename(os.path.join(target, filename), os.path.join(target,filename[:-len(Ext)]+text+Ext))

print('Done.')
