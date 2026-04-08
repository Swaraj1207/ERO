import random
import json
import os

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

class Neuron:
    def __init__(self, num_inputs, use_activation=True):
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(num_inputs)]
        self.bias = 0.0
        self.use_activation = use_activation
        self.last_inputs = []
        self.last_raw = 0.0
        self.last_output = 0.0

    def forward(self, inputs):
        self.last_inputs = inputs
        self.last_raw = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.last_output = relu(self.last_raw) if self.use_activation else self.last_raw
        return self.last_output

    def backward(self, grad_output, learning_rate):
        if self.use_activation:
            grad = grad_output * relu_derivative(self.last_raw)
        else:
            grad = grad_output

        grad_inputs = []
        for i in range(len(self.weights)):
            grad_w = grad * self.last_inputs[i]
            grad_inputs.append(grad * self.weights[i])
            self.weights[i] -= learning_rate * grad_w

        self.bias -= learning_rate * grad
        return grad_inputs

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron, use_activation=True):
        self.neurons = [Neuron(num_inputs_per_neuron, use_activation) for _ in range(num_neurons)]

    def forward(self, inputs):
        return [n.forward(inputs) for n in self.neurons]

    def backward(self, grad_outputs, learning_rate):
        all_grad_inputs = [n.backward(g, learning_rate) for n, g in zip(self.neurons, grad_outputs)]
        num_inputs = len(all_grad_inputs[0])
        combined = []
        for i in range(num_inputs):
            combined.append(sum(g[i] for g in all_grad_inputs))
        return combined

class Network:
    def __init__(self):
        self.layer1 = Layer(num_neurons=16, num_inputs_per_neuron=2, use_activation=True)
        self.layer2 = Layer(num_neurons=16, num_inputs_per_neuron=16, use_activation=True)
        self.layer3 = Layer(num_neurons=1, num_inputs_per_neuron=16, use_activation=False)

    def forward(self, inputs):
        out1 = self.layer1.forward(inputs)
        out2 = self.layer2.forward(out1)
        out3 = self.layer3.forward(out2)
        return out3[0]

    def backward(self, predicted, actual, learning_rate):
        error = predicted - actual
        grad = self.layer3.backward([error], learning_rate)
        grad = self.layer2.backward(grad, learning_rate)
        self.layer1.backward(grad, learning_rate)

def save_model(net, filename):
    data = {}
    for layer_idx, layer in enumerate([net.layer1, net.layer2, net.layer3]):
        for neuron_idx, neuron in enumerate(layer.neurons):
            key = f"L{layer_idx}_N{neuron_idx}"
            data[key] = {"weights": neuron.weights, "bias": neuron.bias}
    with open(filename, "w") as f:
        json.dump(data, f)

def load_model(net, filename):
    with open(filename, "r") as f:
        data = json.load(f)
    for layer_idx, layer in enumerate([net.layer1, net.layer2, net.layer3]):
        for neuron_idx, neuron in enumerate(layer.neurons):
            key = f"L{layer_idx}_N{neuron_idx}"
            neuron.weights = data[key]["weights"]
            neuron.bias = data[key]["bias"]

ops = {
    "add": {
        "file": "brain_add.json",
        "norm_in":  lambda a, b: [a / 9.0, b / 9.0],
        "norm_out": lambda x: x / 18.0,
        "denorm":   lambda x: x * 18.0,
        "epochs": 80000,
        "data": [
            ([1,2], 3), ([3,5], 8), ([2,4], 6),
            ([6,3], 9), ([4,4], 8), ([7,2], 9),
            ([5,5], 10), ([8,1], 9), ([3,3], 6),
            ([9,0], 9), ([1,8], 9), ([4,5], 9),
            ([2,7], 9), ([6,2], 8), ([0,0], 0),
            ([7,7], 14), ([8,8], 16), ([9,8], 17),
            ([6,8], 14), ([7,6], 13), ([8,6], 14),
            ([9,7], 16), ([5,8], 13), ([8,9], 17),
        ]
    },
    "sub": {
        "file": "brain_sub.json",
        "norm_in":  lambda a, b: [a / 9.0, b / 9.0],
        "norm_out": lambda x: (x + 9) / 18.0,
        "denorm":   lambda x: x * 18.0 - 9,
        "epochs": 50000,
        "data": [
            ([5,2], 3), ([9,3], 6), ([8,5], 3),
            ([6,1], 5), ([7,4], 3), ([9,9], 0),
            ([8,3], 5), ([7,2], 5), ([6,4], 2),
            ([9,1], 8), ([5,5], 0), ([4,3], 1),
            ([3,1], 2), ([8,8], 0), ([9,2], 7),
        ]
    },
    "mul": {
        "file": "brain_mul.json",
        "norm_in":  lambda a, b: [a / 9.0, b / 9.0],
        "norm_out": lambda x: x / 81.0,
        "denorm":   lambda x: x * 81.0,
        "epochs": 150000,
        "data": [
            ([2,3], 6), ([3,3], 9), ([4,2], 8),
            ([5,3], 15), ([2,2], 4), ([6,3], 18),
            ([7,2], 14), ([4,4], 16), ([5,5], 25),
            ([3,6], 18), ([9,2], 18), ([8,3], 24),
            ([7,3], 21), ([6,6], 36), ([9,9], 81),
            ([7,8], 56), ([8,7], 56), ([6,7], 42),
            ([7,7], 49), ([8,8], 64), ([9,8], 72),
            ([5,7], 35), ([6,8], 48), ([7,9], 63),
        ]
    },
    "div": {
        "file": "brain_div.json",
        "norm_in":  lambda a, b: [a / 9.0, b / 9.0],
        "norm_out": lambda x: x / 9.0,
        "denorm":   lambda x: x * 9.0,
        "epochs": 50000,
        "data": [
            ([6,2], 3), ([9,3], 3), ([8,4], 2),
            ([4,2], 2), ([6,3], 2), ([9,9], 1),
            ([8,2], 4), ([6,6], 1), ([4,4], 1),
            ([9,1], 9), ([8,8], 1), ([6,1], 6),
            ([4,1], 4), ([9,3], 3), ([8,1], 8),
        ]
    }
}

def train(net, op, learning_rate=0.01):
    epochs = op.get("epochs", 50000)
    for epoch in range(epochs):
        total_loss = 0
        for inputs, actual in op["data"]:
            norm_in = op["norm_in"](inputs[0], inputs[1])
            norm_actual = op["norm_out"](actual)
            predicted = net.forward(norm_in)
            loss = (predicted - norm_actual) ** 2
            total_loss += loss
            net.backward(predicted, norm_actual, learning_rate)
        if epoch % 10000 == 0:
            print(f"  Epoch {epoch} | Loss: {total_loss:.6f}")

# Word maps
word_to_num = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,
    "five":5,"six":6,"seven":7,"eight":8,"nine":9
}

word_to_op = {
    "plus":"add","add":"add","added":"add","sum":"add","and":"add",
    "minus":"sub","subtract":"sub","subtracted":"sub","less":"sub",
    "times":"mul","multiply":"mul","multiplied":"mul","product":"mul","x":"mul",
    "divide":"div","divided":"div","split":"div","over":"div","by":"div"
}

def parse(text):
    tokens = text.lower().replace("?","").replace(",","").split()
    numbers = []
    op = None

    for token in tokens:
        if token in word_to_num:
            numbers.append(float(word_to_num[token]))
        elif token.replace(".","").isdigit():
            numbers.append(float(token))
        if token in word_to_op:
            op = word_to_op[token]

    if len(numbers) == 2 and op:
        return numbers[0], numbers[1], op
    return None

# TRAIN OR LOAD
networks = {}
for name, op in ops.items():
    net = Network()
    if os.path.exists(op["file"]):
        load_model(net, op["file"])
        print(f"{name.upper()} brain loaded ← {op['file']}")
    else:
        print(f"\nTraining {name.upper()} network...")
        train(net, op)
        save_model(net, op["file"])
        print(f"  Saved → {op['file']}")
    networks[name] = net

# INTERACTIVE
print("\n========= YOUR AI IS READY =========")
print("Talk to it naturally!")
print("Examples:")
print("  'what is 3 plus 5'")
print("  'multiply 4 by 3'")
print("  'what is nine minus two'")
print("  'divide 8 by 2'")
print("  '6 + 7' or '9 - 3' or '4 x 3'")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break

    # Try symbol parsing first
    parsed = None
    for sym, op_name in [("+","add"),("-","sub"),("x","mul"),("/","div"),("*","mul")]:
        if sym in user_input:
            try:
                a, b = user_input.split(sym)
                parsed = (float(a.strip()), float(b.strip()), op_name)
                break
            except:
                pass

    # Fall back to natural language
    if not parsed:
        parsed = parse(user_input)

    if parsed:
        a, b, op_name = parsed
        op = ops[op_name]
        result = op["denorm"](networks[op_name].forward(op["norm_in"](a, b)))
        print(f"AI: {round(result, 1)}\n")
    else:
        print("AI: I didn't understand. Try 'what is 3 plus 5'\n")