print("Hello")
animals = ['lion', 'cat', 'tiger']

for id, animal in enumerate(animals):
    print('#%d: %s' % (id+1, animal))

animalLegs = {'lion': 4, 'cat': 4, 'spider': 8}
for animal in animalLegs:
    print('A %s has %d legs' % (animal, animalLegs[animal]))

t = (5,6)
#tuples are immutable t[0] = 7
print(type(t))


class Greeter:

    def __init__(self, name):
        self.name = name

    def hello(self, loud=False):
        hellostring = 'Hello %s' % (self.name)
        if(loud):
            return hellostring.upper()
        else:
            return hellostring

#print(hello('Jakub'))
#print(hello('Ash', True))

g = Greeter('Jakub')
print(Greeter.hello)    # returns function
print(g.hello())
print(g.hello(True))


