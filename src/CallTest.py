class Person:
    def __call__(self, name):
        print("__call__"+"Hello " +name)   #可以不用.xxx直接调用

    def hello(self, name):
        print("hello" + name)              #用.xxx调用


person = Person()
person("zhangsan")
person.hello("lisi")
