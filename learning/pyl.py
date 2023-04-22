# //和**
print(17 / 3)
print(17 // 3)
print(17 % 3)

print(5 ** 2)

# 字符串
print('''\
hello \
world
''')
print('"hello"')
print('\'hello\'')
print('hello\name')
print(r'hello\name')

print('hello'
      'world')
print('hello' + 'world')

# 切片左闭右开
s = 'hello world'
print(s[0:-1])

# 字符串不可修改
# s[0] = 'a'

question = ['a', 'b', 'c']
answer = [1, 2, 3]
for q, a in zip(question, answer):
    print(q, a)
