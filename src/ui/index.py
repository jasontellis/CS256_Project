from flask import Flask, url_for, request
import os
app = Flask(__name__)

HTML_PREFIX = '<!DOCTYPE html>' \
              '<html lang="en">' \
              '<head>' \
              '<title>Enhanced Images</title> ' \
              '<meta charset="utf-8"> ' \
              '<meta name="viewport" content="width=device-width, initial-scale=1"> ' \
              '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"> ' \
              '<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script> ' \
              '<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>' \
              '<style>.' \
              'imgContainer{' \
              'height:40%%;' \
              'position:relative;' \
              'top:40px; ' \
              'display:inline-block; ' \
              'text-align:center; ' \
              '}' \
              '' \
              '.imgButtonLike{' \
              'position:absolute;' \
              'top:80%%;' \
              'left:30%%; ' \
              'width:100px; ' \
              'height:100px;' \
              '}' \
              '' \
              '.imgButtonDislike{' \
              'position:absolute;' \
              'top:50%%;' \
              'right:40%%; ' \
              'width:100px; ' \
              'height:100px;' \
              '}' \
              '</style>' \
              '</head>' \
              '<body>' \
              '<div class="container">'


HTML_SUFFIX = '</div></body></html>'

@app.route('/')
def index():
    return 'Index Page'

@app.route('/authorize/')
def authorize():
	# return 'Hi'
	return app.send_static_file('fbLogin.html')

@app.route('/images/<string:username>/',methods = ['GET', 'POST'])
def images(username):
	return getHTML(username)
    # return 'Hello, World!'

@app.route('/postFeedback/<string:username>/<string:feedback>/',methods = ['GET', 'POST'])
def postFeedback(username, feedback):
    return 'Feedback posted'

def getFileList(username):
	basePath = os.path.dirname(__file__)
	imagesFilePath = os.path.abspath(os.path.join(basePath, 'static', 'images', username))
	print imagesFilePath
	fileList = [(os.path.join(imagesFilePath, file), file) for file in os.listdir(imagesFilePath) if
		            os.path.isfile(os.path.join(imagesFilePath, file)) and (file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"))]

	return fileList

def getHTML(username):
	fileList = getFileList(username)
	carouselBody = generateCarousel(fileList, username)
	html = HTML_PREFIX + carouselBody + HTML_SUFFIX
	return html



def generateCarousel(fileList, username, carouselID = 'myCarousel'):
	prefix = '<div id="%s" class="carousel slide">'%(carouselID)
	suffix = '</div>'
	outer = __generateCarouselInner__(fileList, username)
	inner = __generateCarouselOuter__(fileList, carouselID)
	controls = __getCarouselControls__(carouselID)
	carousel = prefix + outer + inner + controls + suffix

	return carousel

def __generateCarouselOuter__(fileList, carouselID):
	outputString = ''
	prefix = '<ol class="carousel-indicators">'
	suffix = '</ol>'
	activeClass = 'class="active">'

	counter = 0
	for file in fileList:
		counter += 1
		liClass = ''
		if counter == 1:
			liClass = activeClass
	outputString += '<li data-target="#%s" data-slide-to="%s" %s></li>'%(carouselID, (counter-1), liClass)
	outputString = prefix + outputString + suffix

	return outputString

def __generateCarouselInner__(fileList, username):
	outputString = ''
	prefix = '<div class="carousel-inner">'
	suffix = '</div>'
	activeClass = 'item active'

	counter = 0
	for file in fileList:
		relPath = os.path.relpath(file[0])
		imageURL = url_for('static', filename='images/%s/%s'%(username, file[1]))
		print relPath
		counter += 1
		divClass = 'item'
		if counter == 1:
			divClass = activeClass
		imageString = '<div class="%s imgContainer thumbnail" background-image=%s>' \
		              '<button class="btn-success imgButtonLike" ' \
		              'style ="position: absolute; top:40%%; left: 30%%;">' \
		              '<span style="font-size:1.5em;" class="glyphicon glyphicon-ok" aria-hidden="true"></span>' \
		              'Like' \
		              '</button>' \
		              '' \
		              '<button class="btn-danger imgButtonDislike" style ="position: absolute; top:40%%; right: 30%%;">' \
		              'Dislike  ' \
		              '<span style="font-size:1.5em;" class="glyphicon glyphicon-remove" aria-hidden="true"></span>' \
		              '</button>' \
		              '<img  class="img-rounded img-responsive" src=%s style="max-height: 750px;">' \
		              '</div>'%(divClass, imageURL, imageURL)
		outputString  += imageString

	outputString = prefix + outputString + suffix

	return outputString

def __getCarouselControls__(carouselID):

	controls = '<a class="left carousel-control" href="#%s" data-slide="prev"> ' \
	           '<span class="glyphicon glyphicon-chevron-left"></span> ' \
	           '<span class="sr-only">Previous</span> ' \
	           '</a> ' \
	           '' \
	           '<a class="right carousel-control" href="#%s" data-slide="next"> ' \
	           '<span class="glyphicon glyphicon-chevron-right"></span> ' \
	           '<span class="sr-only">Next</span> ' \
	           '</a>'%(carouselID, carouselID)
	return controls

if __name__ == '__main__':
	print getHTML('2')

