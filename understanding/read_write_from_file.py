#------------
file = open('write_file.txt', 'w')
file.write('Hello World\n')
file.write('Hello World\n')
file.write('Hello World\n')
#------------
# use open to open a file 
# 'w' is for write mode (create a new file or overwrite an existing file)



#------------
file = open('write_file.txt', 'r')
print(file.read())
file.close()
#------------
# use open to open a file
# 'r' is for read mode (read an existing file and if it does not exist, it will throw an error)

#------------
file = open('write_file.txt', 'r')
for line in file:
    print(line.strip())
file.close()
#------------
# use open to open a file
# 'r' is for read mode (read an existing file and if it does not exist, it will throw an error)
# for line in file: (is used to read the file line by line)


#------------
with open('write_file.txt', 'r') as file:
    for line in file:
        print(line.strip())
#------------
# use open to open a file
# 'r' is for read mode (read an existing file and if it does not exist, it will throw an error)
#with use statement, the file is automatically closed after the block of code is executed