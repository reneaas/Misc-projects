import rect

x0, y0, x1, y1 = 1, 2, 3, 4

rect_obj = rect.PyRectangle(x0, y0, x1, y1)
print(dir(rect_obj))
print(rect_obj.get_area())
print(rect_obj.get_size())
del rect_obj
