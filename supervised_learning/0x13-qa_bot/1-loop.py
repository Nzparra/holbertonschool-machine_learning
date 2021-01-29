  
#!/usr/bin/env python3
""" takes in input from the user """

while 1:
    prompt = input("Q: ")
    if prompt.lower() in ['exit', 'goodbye', 'bye']:
        print("A: Goodbye")
        exit(0)
    else:
        print("A:")