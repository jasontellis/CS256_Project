import facebook


def getImages(id, token):
	graph = facebook.GraphAPI(access_token=token)
	images = graph.get_connections(id=id, connection_name='photos', fields='id,name,picture,images,link,source')
	print len(images['data']),'\n',images,
	# images = graph.get_object(id=id, fields=''))

if __name__ == '__main__':
	id =    '10152457319305628'
	token = 'EAACEdEose0cBAOnGqUXwDYxZAP9wazWoVK4yPDBdGZAyKr65SCnCV00tVijhJDM8nrB697ZCrkowshe8dE1J9oESR52ZBQ8Xo7AICaI2iSG8vrp4xIpodHCVkpEu87Mbt3bUbHOy9N4YOFrKTtlfcLirnbmbzZC0eQtDQxg5XbJkzUrIda0c2aMX8Ba8NW9gZD'
	getImages(id,token)
