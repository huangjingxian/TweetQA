import json

text = []
train,dev,test = json.load(open("com_train.json")),json.load(open("com_dev.json")),json.load(open("com_test.json"))

for i in range(len(train)):
	query = train[i]
	text.append(query['Tweet'])
	text.append(query['Question'])
	text.append(query['Answer'][0])

for i in range(len(dev)):
	query = dev[i]
	text.append(query['Tweet'])
	text.append(query['Question'])
	text.append(query['Answer'][0])
	if query['Answer'][1] != query['Answer'][0]:
		text.append(query['Answer'][1])

for i in range(len(train)):
	query = train[i]
	text.append(query['Tweet'])
	text.append(query['Question'])

text8 = ' '.join(text)

text_file = open("text8", "w")
text_file.write(text8)
text_file.close()