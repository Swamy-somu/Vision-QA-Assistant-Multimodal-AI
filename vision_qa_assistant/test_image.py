from PIL import Image

img = Image.open("samples/apple.jpg")
print("Image format:", img.format)
print("Image size:", img.size)
img.show()

