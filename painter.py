import tkinter as tk
import numpy as np
import network
import mnist_loader

class DigitDrawingApp:
    def __init__(self, root, net):
        self.net = net
        self.root = root
        self.cell_size = 20  # Each cell is 20x20 pixels
        self.grid_size = 28  # 28x28 grid

        self.canvas = tk.Canvas(root, width=self.grid_size * self.cell_size, height=self.grid_size * self.cell_size, bg="white")
        self.canvas.pack()

        self.drawing_data = np.zeros((self.grid_size, self.grid_size))  # 28x28 matrix to store pixel values

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.is_drawing = False
        self.last_x, self.last_y = None, None

        # Add buttons
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.export_button = tk.Button(root, text="Export", command=self.export_drawing)
        self.export_button.pack(side=tk.LEFT)

        self.result_label = tk.Label(root, text="Draw a digit and click Export")
        self.result_label.pack(side=tk.LEFT)

    def draw(self, event):
        # Convert mouse position to grid cell
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if 0 <= col < self.grid_size and 0 <= row < self.grid_size:
            # Fill the cell on the canvas
            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")

            # Update the drawing data
            self.drawing_data[row, col] = 1

    def stop_drawing(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing_data.fill(0)  # Reset the data matrix
        self.result_label.config(text="Draw a digit and click Export")

    def export_drawing(self):
        # Flatten the 28x28 matrix to a 784-element vector and pass it to the neural network
        input_data = self.drawing_data.flatten().reshape(-1, 1)
        recognized_digit = np.argmax(self.net.feedforward(input_data))
        self.result_label.config(text=f"Recognized Digit: {recognized_digit}")

if __name__ == "__main__":
    # Load data and train the network
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.train(training_data, 15, 10, 3, test_data)

    root = tk.Tk()
    app = DigitDrawingApp(root, net)
    root.mainloop()
