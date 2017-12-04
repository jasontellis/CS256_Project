import facebook


def getImages(id, token):
	graph = facebook.GraphAPI(access_token=token)
	images = graph.get_connections(id=id, connection_name='photos', fields='id,name,picture,images,link,source')
	print len(images['data']),'\n',images,
	# images = graph.get_object(id=id, fields=''))

if __name__ == '__main__':
	id =    '10152457319305628'
	token = 'EAACEdEose0cBAPZACZBPFhHatkrZC4wkp4qDjTnQWAgrzJFLyDSy8mW4mJoiS783nMZAbNRFBKEhuUjObbtn4f3ugSRZCtX4hyv1TXhCBmqhuXxwV6vPB3lzIMVYOZBJQmCTjSpX4DZCz7Ob9Rc4iv4iNVpYgsixEaQlFZCler3R50PPkWvlkrxIuaJVxjZAi2hgZD'
	getImages(id,token)
