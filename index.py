salt = "Ütvefúró"
source = "Have a nice day hacker"

def enCrypt(data,salt): 
    hayStack = ''.join(format(ord(i), '08b') for i in data)
    needle  =  ''.join(format(ord(i), '08b') for i in salt)
    output = ""
    for index, char in enumerate(hayStack):
        if index < len(needle) :
            output += str(int(hayStack[index]) + int(needle[index]))
        else:
            output += str(hayStack[index])
    return output

def deCrypt(data,salt):
    tmp = ""
    needle  =  ''.join(format(ord(i), '08b') for i in salt)
    for index, char in enumerate(data):
        if index < len(needle) :
            tmp += str(int(data[index]) - int(needle[index]))
        else:
            tmp += str(data[index])
    tmp = ' '.join([tmp[i:i+8] for i in range(0, len(tmp), 8)])
    binary_values = tmp.split(' ')
    text = ''.join(chr(int(b, 2)) for b in binary_values)
    return text

encryptedString = enCrypt(source,salt)
print( "This is the encrypted string: " + encryptedString )


decryptedString = deCrypt(encryptedString,salt) #valid salt
#decryptedString = deCrypt(encryptedString,"HackThis")  #invalid salt

print("This is the decrypted string: "+ decryptedString )




