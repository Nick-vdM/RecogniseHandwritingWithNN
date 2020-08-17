# This file should do all of the maths for me!
import math as m

def fakeDot(w1, i1, w3, i2, b1, w11):
    return w1 * i1 + w3 * i2 + b1 * w11

def sigmoid(value):
    return 1 / ( 1 + m.exp(-value) )

def errorSquared(target, output):
    return 1/2 * (target - output) ** 2
    
def devE_devOut(target, out):
    return (target - out) * -1

def devOut_devNet(out):
    return out * ( 1 - out )

w1 = 0.1
w2 = 0.2
w3 = 0.1
w4 = 0.1
w5 = 0.1
w6 = 0.1
w7 = 0.1
w8 = 0.2
w9 = 0.1
w10 = 0.1
w11 = 0.1
w12 = 0.1

print("===================== First Round =========================")
print("---------------------- Forwards ---------------------------")
one_i1 = 0.1
one_i2 = 0.2
one_o1 = 0
one_o2 = 1

one_net_h1 = fakeDot(w1, one_i1, w3, one_i2, w9, 1)
print("one_net_h1 =", one_net_h1)
one_out_h1 = sigmoid(one_net_h1)
print("one_out_h1 =", one_out_h1)
print() # newline

one_net_h2 = fakeDot(w2, one_i1, w4, one_i2, w10, 1)
print("one_net_h2 =", one_net_h2)
one_out_h2 = sigmoid(one_net_h2)
print("one_out_h2 =", one_out_h2)
print() # newline

one_net_o1 = fakeDot(w5, one_out_h1, w7, one_out_h2, w11, 1)
print("one_net_o1 =", one_net_o1)
one_out_o1 = sigmoid(one_net_o1)
print("one_out_o1 =", one_out_o1)
print() # newline

one_net_o2 = fakeDot(w6, one_out_h1, w8, one_out_h2, w12, 1)
print("one_net_o2 =", one_net_o2)
one_out_o2 = sigmoid(one_net_o2)
print("one_out_o2 =", one_out_o2)
print() # newline

print("---------------------- Error ---------------------------")
one_error_o1 = errorSquared(one_o1, one_out_o1)
print("Error 1 =", one_error_o1)
one_error_o2 = errorSquared(one_o2, one_out_o2)
print("Error 2 =", one_error_o2)
one_error_total = one_error_o1 + one_error_o2
print("Total error =", one_error_total)

print("------------------- Backwards --------------------------")
print("ZZZZZZZZZZZZZZZZZZ First Layer ZZZZZZZZZZZZZZZZZZZZZZZZZ")
print("- - - - - - - - - -  Weight 5 - - - - - - - - - - - - -")
one_devE_devOuto1 = devE_devOut(one_o1, one_out_o1)
print("devE / devOutO1", one_devE_devOuto1)
one_devOuto1_devNeto1 = devOut_devNet(one_out_o1)
print("devOuto1 / devNeto1", one_devOuto1_devNeto1)
one_devNeto1_devW5 = one_out_h1
print("devNeto1 / devW5", one_devNeto1_devW5)
one_devE_devW5 = one_devE_devOuto1 * one_devOuto1_devNeto1 * one_devNeto1_devW5
print("dev E / w5", one_devE_devW5)

print("- - - - - - - - - - Weight 7 - - - - - - - - - - - - - - -")
one_devNeto1_devW7 = one_out_h2
print("devNeto1_devW7 =", one_devNeto1_devW7)
one_devE_devW7 = one_devE_devOuto1 * one_devOuto1_devNeto1 * one_devNeto1_devW7
print("devE_devW7 =", one_devE_devW7)


print("- - - - - - - - - -  Weight 6 - - - - - - - - - - - - -")
one_devE_devOuto2 = devE_devOut(one_o2, one_out_o2)
print("devE / devOutO2", one_devE_devOuto2)
one_devOuto2_devNeto2 = devOut_devNet(one_out_o2)
print("devOuto2 / devNeto2", one_devOuto2_devNeto2)
one_devNeto1_devW6 = one_out_h1
print("devNeto2 / devW6", one_devNeto1_devW6)
one_devE_devW6 = one_devE_devOuto2 * one_devOuto2_devNeto2 * one_devNeto1_devW6
print("dev E / w6", one_devE_devW6)

print("- - - - - - - - - - Weight 8 - - - - - - - - - - - - - - -")
one_devNeto1_devW8 = one_out_h2
print("devNeto1_devW8 =", one_devNeto1_devW8)
one_devE_devW8 = one_devE_devOuto2 * one_devOuto2_devNeto2 * one_devNeto1_devW8
print("devE / devW8 =", one_devE_devW8)
#TODO: Figure out how to calculate the biases

# now for the... fun part...
print("ZZZZZZZZZZZZZZZZZZ Second Layer ZZZZZZZZZZZZZZZZZZZZZZZZZ")
print("- - - - - - - - - - Weight 1 - - - - - - - - - - - - - - -")
# This equation is usually devEO1 not devETotal but in this case they're the 
# same
one_devEo1_devNeth1 = one_devE_devOuto1 * one_devOuto1_devNeto1
print("devEo1 / devNeth1 =", one_devEo1_devNeth1)
one_devOuto1_devNeth1 = w5
print("devOuto1 / devNeth1 =", one_devOuto1_devNeth1)
one_devEo1_devOuth1 = one_devEo1_devNeth1 * one_devOuto1_devNeth1
print("one_devEo1_devOuth1 =", one_devEo1_devOuth1)

print("- - - - - - - - - - Weight 2 - - - - - - - - - - - - - - -")
print("- - - - - - - - - - Weight 3 - - - - - - - - - - - - - - -")
print("- - - - - - - - - - Weight 4 - - - - - - - - - - - - - - -")
