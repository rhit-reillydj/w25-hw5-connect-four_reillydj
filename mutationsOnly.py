import copy
from multiprocessing import Pool

import random
import numpy as np
import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F


POPULATION_SIZE = 50
GENERATIONS = 1000
BATCH_SIZE = 5


def create_board():
    """
    Create an empty Connect Four board with -2 padding around the borders.
    Returns:
        torch.Tensor: A padded 2D tensor of shape (1, 1, 8, 9).
    """
    board = torch.zeros((1, 1, 6, 7), dtype=torch.float32)
    padded_board = F.pad(board, pad=(1, 1, 1, 1), mode='constant', value=-2)
    return padded_board


def print_board(board):
    """
    Print the Connect Four board in a human-readable format.
    
    Args:
        board (torch.Tensor): A 2D tensor representing the game board.
    """
    # Map numbers to symbols for display
    symbol_map = {0: ' ', 1: 'R', -1: 'Y'}
    print('\n 1 2 3 4 5 6 7')
    print('---------------')
    for row in board:
        print('|' + '|'.join(symbol_map[cell.item()] for cell in row) + '|')
    print('---------------')


def is_valid_move(board, col):
    """
    Check if a move is valid (column is not full).
    
    Args:
        board (torch.Tensor): A 2D tensor representing the game board.
        col (int): The column index (0-6).
    
    Returns:
        bool: True if the move is valid, False otherwise.
    """
    return 0 <= col <= 6 and board[0, col] == 0


def drop_piece(board, col, value):
    """
    Drop a piece into the board in the specified column.
    
    Args:
        board (torch.Tensor): A 2D tensor representing the game board.
        col (int): The column index (0-6).
        value (int): The value of the piece to drop (1 for Red, -1 for Yellow).
    
    Returns:
        int: The row index where the piece was dropped, or -1 if the column is full.
    """
    for row in range(5, -1, -1):
        if board[row, col] == 0:
            board[row, col] = value
            return row
    return -1


def check_win(board, last_row, last_col, value):
    """
    Check if a move results in a win for the current player.
    
    Args:
        board (torch.Tensor): A 2D tensor representing the game board.
        last_row (int): The row index of the last move.
        last_col (int): The column index of the last move.
        value (int): The value of the current player (1 for Red, -1 for Yellow).
    
    Returns:
        bool: True if the player wins, False otherwise.
    """
    directions = [
        [(0, 1), (0, -1)],  # Horizontal
        [(1, 0), (-1, 0)],  # Vertical
        [(1, 1), (-1, -1)], # Diagonal positive
        [(1, -1), (-1, 1)]  # Diagonal negative
    ]
    
    def count_direction(row, col, row_step, col_step):
        count = 0
        r, c = row, col
        
        while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == value and count < 4:
            count += 1
            r += row_step
            c += col_step
        return count
    
    for direction_pair in directions:
        total_count = -1  # -1 because we count the last move twice
        for row_step, col_step in direction_pair:
            count = count_direction(last_row, last_col, row_step, col_step)
            total_count += count
            if total_count >= 4:
                return True
    return False


def play_game():
    """
    Play a game of Connect Four with the board represented as a PyTorch tensor.
    """
    board = create_board()
    current_value = 1  # 1 for Red, -1 for Yellow
    moves_made = 0
    
    while True:
        print_board(board)
        player_name = "Red" if current_value == 1 else "Yellow"
        
        while True:
            try:
                col = int(input(f"\n{player_name}'s turn (1-7): ")) - 1
                if is_valid_move(board, col):
                    break
                print("Invalid move. Column is either full or out of range.")
            except ValueError:
                print("Please enter a number between 1 and 7.")
        
        row = drop_piece(board, col, current_value)
        moves_made += 1
        
        if moves_made >= 7 and check_win(board, row, col, current_value):
            print_board(board)
            print(f"\n{player_name} wins!")
            break
        
        if moves_made == 42:
            print_board(board)
            print("\nIt's a draw!")
            break
        
        current_value *= -1  # Alternate between 1 and -1
        
        
        
        
def create_random_boards(max_pieces=20, batch_size=BATCH_SIZE):
    """
    Create a batch of random Connect Four boards with valid piece placements.
    Ensure it is always Red's turn to play. Include sparse and empty boards.

    Args:
        max_pieces (int): Maximum number of pieces to place on a single board.
    
    Returns:
        torch.Tensor: A tensor of shape (BATCH_SIZE, 1, 6, 7) representing the boards.
    """
    boards = torch.zeros((batch_size, 1, 6, 7), dtype=torch.float32)

    for b in range(batch_size):
        # Randomly decide the number of pieces, ensuring it's even
        total_pieces = random.choice([0, 2, 4, 6, 8, random.randint(10, max_pieces)])  # Sparse and varied boards
        red_count = total_pieces // 2  # Equal Red and Yellow pieces for Red's turn
        yellow_count = total_pieces // 2

        for _ in range(total_pieces):
            color = 1 if red_count > yellow_count else -1  # Place Red first for balance
            red_count -= color == 1
            yellow_count -= color == -1

            while True:
                col = random.randint(0, 6)  # Random column
                for row in range(5, -1, -1):  # Find the first empty row from the bottom
                    if boards[b, 0, row, col] == 0:
                        boards[b, 0, row, col] = color
                        break
                else:
                    continue  # Column is full, pick another
                break

    return boards




def heuristic_evaluate_batch(boards, player_value=1):
    """
    Heuristically evaluate a batch of Connect Four boards.
    Scores range from -1 (Yellow dominating) to 1 (Red dominating), with additional
    value for control of the center of the board.
    
    Args:
        boards (torch.Tensor): A tensor of shape (BATCH_SIZE, 1, 6, 7) representing the boards.
        player_value (int): 1 for Red, -1 for Yellow (the player to evaluate for).
    
    Returns:
        torch.Tensor: A tensor of shape (BATCH_SIZE,) with heuristic scores for each board.
    """
    scores = torch.zeros(BATCH_SIZE, dtype=torch.float32)

    directions = [
        [(0, 1), (0, -1)],  # Horizontal
        [(1, 0), (-1, 0)],  # Vertical
        [(1, 1), (-1, -1)], # Diagonal positive
        [(1, -1), (-1, 1)]  # Diagonal negative
    ]

    # Central column weights (heavier weights for central columns)
    center_weights = torch.tensor([1, 2, 3, 4, 3, 2, 1], dtype=torch.float32)

    def count_consecutive(board, row, col, value, row_step, col_step):
        """
        Count consecutive pieces in a given direction on a single board.
        """
        count = 0
        r, c = row, col
        while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == value:
            count += 1
            r += row_step
            c += col_step
        return count

    for b in range(BATCH_SIZE):
        board = boards[b, 0]  # Extract a single board

        for r in range(6):
            for c in range(7):
                if board[r, c] == 0:
                    continue  # Skip empty cells
                
                current_value = board[r, c]

                for direction in directions:
                    # Check both directions for a potential connection
                    total_count = -1  # To avoid double-counting the current cell
                    for row_step, col_step in direction:
                        total_count += count_consecutive(board, r, c, current_value, row_step, col_step)
                    
                    # Score based on the length of the connection
                    if total_count >= 4:  # Winning position
                        if current_value == player_value:
                            scores[b] = 1  # Immediate win for the player
                            break
                        else:
                            scores[b] = -1  # Immediate win for the opponent
                            break
                    elif total_count == 3:  # Strong connection
                        scores[b] += 0.5 * (1 if current_value == player_value else -1)
                    elif total_count == 2:  # Potential connection
                        scores[b] += 0.2 * (1 if current_value == player_value else -1)

        # Add center control value
        for c in range(7):
            column_control = board[:, c].sum().item()  # Sum of pieces in the column
            scores[b] += center_weights[c] * column_control * 0.1  # Scale the contribution of center control

    # Normalize scores to range [-1, 1]
    scores = torch.clamp(scores, -1, 1)
    return scores


def select_best_available_column(output_scores, board):
    """
    Select the best available column from the output scores.

    Args:
        output_scores (torch.Tensor): The output scores for each column.
        board (torch.Tensor): The current board.

    Returns:
        int: The best available column index.
    """
    sorted_columns = torch.argsort(output_scores, descending=True)
    for col in sorted_columns:
        col = col.item()
        if is_valid_move(board, col):
            return col
    return -1  # This should not happen unless all columns are full




# if __name__ == "__main__":
#     print("Welcome to Connect Four!")
#     play_game()

    
























# AI Part
class CNN:
    def __init__(self, genome=None, input_size=(6, 7), input_channels=1, fixed_output_channels=32):
        self.fixed_output_channels = fixed_output_channels  # Fixed output channels for FFN
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size is (1, 1)

        if genome:
            self.genome = genome
        else:
            current_height, current_width = input_size
            current_channels = input_channels

            self.genome = {"cnn_layers": [], "ffn_layers": []}

            # Define CNN layers
            cnn_layers = [
                {"type": "conv", "filters": 8, "kernel_size": 3, "stride": 1, "activation": "relu"},
                {"type": "pool", "size": 2, "stride": 2, "pool_type": "max"},
            ]

            # Process CNN layers
            for layer in cnn_layers:
                if layer["type"] == "conv":
                    # Compute output size
                    output_height = compute_conv_output_size(
                        current_height, layer["kernel_size"]
                    )
                    output_width = compute_conv_output_size(
                        current_width, layer["kernel_size"]
                    )

                    layer["weights"] = np.random.uniform(
                        -1, 1, size=(layer["filters"], current_channels, layer["kernel_size"], layer["kernel_size"])
                    )
                    layer["biases"] = np.random.uniform(-1, 1, size=(layer["filters"]))

                    # Update current dimensions
                    current_height, current_width = output_height, output_width
                    current_channels = layer["filters"]

                elif layer["type"] == "pool":
                    # Compute output size after pooling
                    output_height = compute_pool_output_size(current_height, layer["size"])
                    output_width = compute_pool_output_size(current_width, layer["size"])
                    current_height, current_width = output_height, output_width

                self.genome["cnn_layers"].append(layer)

            # Compute the flattened size of the CNN output
            flattened_size = current_height * current_width * current_channels

            # Define FFN layers
            ffn_layers = [
                {"neurons": 32, "activation": "relu"},
                {"neurons": 7, "activation": "softmax"},
            ]

            previous_size = flattened_size
            for layer in ffn_layers:
                layer["weights"] = np.random.uniform(-1, 1, size=(layer["neurons"], previous_size))
                layer["biases"] = np.random.uniform(-1, 1, size=(layer["neurons"]))
                previous_size = layer["neurons"]
                self.genome["ffn_layers"].append(layer)





    def forward_pass(self, input_tensor):
        """
        Perform a forward pass on a batch of input tensors.
        Args:
            input_tensor (torch.Tensor): Tensor of shape (batch_size, 1, 6, 7).
        Returns:
            torch.Tensor: Output tensor after the forward pass.
            torch.Tensor: Predicted column indices for each input in the batch.
        """
        x = input_tensor

        # Process CNN layers
        for layer in self.genome["cnn_layers"]:
            if layer["type"] == "conv":
                # Apply convolutional logic
                conv = nn.Conv2d(
                    in_channels=x.shape[1],
                    out_channels=layer["filters"],
                    kernel_size=(layer["kernel_size"], layer["kernel_size"]),
                    stride=layer.get("stride", 1)
                )
                conv.weight.data = torch.tensor(layer["weights"], dtype=torch.float32)
                conv.bias.data = torch.tensor(layer["biases"], dtype=torch.float32)
                x = conv(x)
                if layer["activation"] == "relu":
                    x = F.relu(x)

            elif layer["type"] == "pool":
                # Apply pooling logic
                pool = nn.MaxPool2d(kernel_size=layer["size"], stride=layer["stride"]) if layer["pool_type"] == "max" else nn.AvgPool2d(kernel_size=layer["size"], stride=layer["stride"])
                x = pool(x)

        # Apply adaptive average pooling
        x = self.adaptive_pool(x)  # Shape becomes (batch_size, num_channels, 1, 1)

        # Flatten for FFN
        x = torch.flatten(x, start_dim=1)  # Shape becomes (batch_size, num_channels)

        # Process FFN layers
        for layer in self.genome["ffn_layers"]:
            dense = nn.Linear(in_features=x.shape[1], out_features=layer["neurons"])
            dense.weight.data = torch.tensor(layer["weights"], dtype=torch.float32)
            dense.bias.data = torch.tensor(layer["biases"], dtype=torch.float32)
            x = dense(x)

        # The final output from the FFN (scores for each column)
        output = x  # Shape: (batch_size, 7), where 7 is the number of columns

        # Predict columns, prioritizing valid moves
        predicted_columns = [
            select_best_available_column(output[i], input_tensor[i, 0]) for i in range(output.size(0))
        ]

        return output, torch.tensor(predicted_columns)


    
    def evaluate_fitness(self, shared_boards):
        """
        Evaluate the CNN's performance by:
        1) Using the provided shared boards.
        2) Evaluating each board's position before a move using `heuristic_evaluate_batch`.
        3) Predicting moves and simulating them on each board in the batch.
        4) Evaluating the new positions after the moves using `heuristic_evaluate_batch`.
        5) Calculating the fitness as the difference between evaluations before and after the moves.

        Args:
            shared_boards (torch.Tensor): The shared random boards for evaluation.

        Returns:
            float: The fitness score of the CNN.
        """
        # Evaluate positions before making a move
        pre_move_evaluations = heuristic_evaluate_batch(shared_boards, player_value=1)

        # Clone boards for simulating moves
        simulated_boards = shared_boards.clone()

        # Predict moves and apply them
        for i in range(BATCH_SIZE):
            board = simulated_boards[i]  # Extract a single board (shape: (1, 6, 7))
            output, _ = self.forward_pass(board.unsqueeze(0))  # Add batch dimension for forward pass
            predicted_col = select_best_available_column(output[0], board[0])  # Use helper function for valid column

            # Make the predicted move
            for row in range(5, -1, -1):  # Start from the bottom row
                if board[0, row, predicted_col] == 0:
                    board[0, row, predicted_col] = 1  # Assume Red is playing
                    break

        # Evaluate positions after making the moves
        post_move_evaluations = heuristic_evaluate_batch(simulated_boards, player_value=1)

        # Calculate fitness as the average improvement across the batch
        fitness_improvements = post_move_evaluations - pre_move_evaluations
        fitness = fitness_improvements.mean().item()

        return fitness







    
    def mutate(self, generation=0):
        """
        Apply mutations with an adaptive rate based on the generation number.
        
        Args:
            generation (int): Current generation number.
        """
        # Base mutation weights
        base_mutation_weights = {
            "add_conv_layer": 0.05,
            "remove_layer": 0.02,
            "modify_layer_params": 0.40,
            "alter_weights_biases": 0.40,
            "add_fnn_neurons": 0.03,
            "remove_fnn_neurons": 0.01,
        }

        # Decay factor for mutation probabilities
        decay = 1 / (1 + 0.05 * generation)  # Adjust decay rate as needed

        # Apply decay to mutation probabilities
        mutation_weights = {k: v * decay for k, v in base_mutation_weights.items()}

        # Normalize weights to sum to 1
        total_weight = sum(mutation_weights.values())
        if total_weight == 0:
            # Prevent division by zero
            mutation_probs = {key: 1/len(mutation_weights) for key in mutation_weights}
        else:
            mutation_probs = {key: val / total_weight for key, val in mutation_weights.items()}

        # Determine the number of mutations to apply
        num_mutations = random.choices([1, 2], weights=[0.9, 0.1], k=1)[0]  # Mostly 1 mutation

        for _ in range(num_mutations):
            # Choose a mutation type based on weights
            mutation_type = random.choices(
                population=list(mutation_probs.keys()),
                weights=list(mutation_probs.values()),
                k=1
            )[0]

            # Apply the chosen mutation
            if mutation_type == "add_conv_layer":
                self.add_layer()
            elif mutation_type == "remove_layer":
                self.remove_layer()
            elif mutation_type == "modify_layer_params":
                self.modify_layer()
            elif mutation_type == "alter_weights_biases":
                self.alter_weights_biases()
            elif mutation_type == "add_fnn_neurons":
                self.add_fnn_neurons()
            elif mutation_type == "remove_fnn_neurons":
                self.remove_ffn_neurons()

        # Ensure sizes and parameters remain valid
        self.recalculate_sizes()
        return self
    
    def copy(self):
        # Use deepcopy to ensure the entire object is copied without shared references
        return copy.deepcopy(self)


    def add_layer(self, max_conv_layers=5, max_pool_layers=3):
        """
        Add a new convolutional or pooling layer to the genome while ensuring the total number of
        convolutional and pooling layers does not exceed specified limits.

        Args:
            max_conv_layers (int): Maximum allowed convolutional layers.
            max_pool_layers (int): Maximum allowed pooling layers.
        """
        # Count current conv and pool layers
        conv_layer_count = sum(1 for layer in self.genome["cnn_layers"] if layer["type"] == "conv")
        pool_layer_count = sum(1 for layer in self.genome["cnn_layers"] if layer["type"] == "pool")

        # Determine the type of layer to add
        if conv_layer_count >= max_conv_layers and pool_layer_count >= max_pool_layers:
            # Abort if both limits are reached
            return
        elif conv_layer_count >= max_conv_layers:
            layer_type = "pool"  # Force pool if conv limit is reached
        elif pool_layer_count >= max_pool_layers:
            layer_type = "conv"  # Force conv if pool limit is reached
        else:
            # Randomly select the layer type if limits are not reached
            layer_type = random.choice(["conv", "pool"])

        if layer_type == "conv":
            # Add a convolutional layer
            filters = random.randint(2, 10)  # Random number of filters 2:10
            kernel_size = random.choice([1, 3, 5])  # Random kernel size

            new_layer = {
                "type": "conv",
                "filters": filters,
                "kernel_size": kernel_size,
                "kernel_size": kernel_size,
                "activation": random.choice(["relu", "sigmoid", "tanh"]),
                "weights": np.random.uniform(-1, 1, size=(filters, 1, kernel_size, kernel_size)),
                "biases": np.random.uniform(-1, 1, size=(filters)),
            }
            
        else:
            # Add a pooling layer
            pool_size = 2
            stride = 2
            pool_type = random.choice(["max", "avg"])

            new_layer = {
                "type": "pool",
                "size": pool_size,
                "stride": stride,
                "pool_type": pool_type,
            }

        # Insert the layer at a random position (not first)
        position = random.randint(1, len(self.genome["cnn_layers"]))
        self.genome["cnn_layers"].insert(position, new_layer)

    def remove_layer(self):
        """Remove a random layer from the genome."""
        if len(self.genome["cnn_layers"]) > 1:  # Ensure there's at least two layer left
            position = random.randint(1, len(self.genome["cnn_layers"]) - 1)
            del self.genome["cnn_layers"][position]

    def modify_layer(self):
        """Modify the parameters of a random layer in the genome."""
        if len(self.genome["cnn_layers"]) > 0:
            layer = random.choice(self.genome["cnn_layers"])
            
            if layer["type"] == "conv":
                # Randomly modify parameters
                filters = random.randint(2, 10)
                kernel_size = random.choice([1, 3, 5])
                activation = random.choice(["relu", "sigmoid", "tanh"])
                
                # Update layer parameters
                layer["filters"] = filters
                layer["kernel_size"] = kernel_size
                layer["activation"] = activation
            
            elif layer["type"] == "pool":
                layer["pool_type"] = random.choice(["max", "avg"])


    def alter_weights_biases(self):
        """Apply small random perturbations to weights and biases."""
        for layer in self.genome["cnn_layers"]:
            if "weights" in layer:
                layer["weights"] += np.random.uniform(-0.1, 0.1, size=layer["weights"].shape)
            if "biases" in layer:
                layer["biases"] += np.random.uniform(-0.1, 0.1, size=layer["biases"].shape)
                
        for layer in self.genome["ffn_layers"]:
            layer["weights"] += np.random.uniform(-0.1, 0.1, size=layer["weights"].shape)
            layer["biases"] += np.random.uniform(-0.1, 0.1, size=layer["biases"].shape)

    def add_fnn_neurons(self):
        """Add neurons to a random hidden fully connected layer."""
        # Ensure there are hidden layers to modify
        if len(self.genome["ffn_layers"]) > 2:  # Exclude input and output layer
            # Randomly choose a hidden layer (excluding the output layer)
            layer_index = random.randint(1, len(self.genome["ffn_layers"]) - 2)
            layer = self.genome["ffn_layers"][layer_index]
            
            # Add neurons
            neurons_to_add = random.randint(1, 16)
            new_weights = np.random.uniform(-1, 1, size=(neurons_to_add, layer["weights"].shape[1]))
            new_biases = np.random.uniform(-1, 1, size=(neurons_to_add))
            layer["weights"] = np.vstack([layer["weights"], new_weights])
            layer["biases"] = np.concatenate([layer["biases"], new_biases])
            layer["neurons"] += neurons_to_add


    def remove_ffn_neurons(self):
        """Remove neurons from a random hidden fully connected layer."""
        # Ensure there are hidden layers to modify
        if len(self.genome["ffn_layers"]) > 1:  # Exclude the output layer
            # Randomly choose a hidden layer (excluding the output layer)
            layer_index = random.randint(1, len(self.genome["ffn_layers"]) - 2)
            layer = self.genome["ffn_layers"][layer_index]
            
            # Remove neurons
            if layer["neurons"] > 1:  # Ensure at least one neuron remains
                neurons_to_remove = random.randint(1, min(16, layer["neurons"] - 1))
                layer["weights"] = layer["weights"][:-neurons_to_remove, :]
                layer["biases"] = layer["biases"][:-neurons_to_remove]
                layer["neurons"] -= neurons_to_remove
                
    def recalculate_sizes(self, input_size=(6, 7), input_channels=1):
        """
        Recalculate the sizes of all layers in the genome and update weights and biases.
        Args:
            input_size (tuple): Initial input size (height, width).
            input_channels (int): Number of input channels.
        """
        current_height, current_width = input_size
        current_channels = input_channels

        for layer in self.genome["cnn_layers"]:
            if layer["type"] == "conv":
                # Adjust kernel size if it's too large
                kh = layer["kernel_size"]
                kw = layer["kernel_size"]
                if kh > current_height:
                    kh = current_height
                if kw > current_width:
                    kw = current_width
                layer["kernel_size"], layer["kernel_size"] = kh, kw

                # Recalculate output size
                output_height = compute_conv_output_size(current_height, kh, stride=layer.get("stride", 1))
                output_width = compute_conv_output_size(current_width, kw, stride=layer.get("stride", 1))

                # Update weights and biases
                layer["weights"] = np.random.uniform(
                    -1, 1, size=(layer["filters"], current_channels, kh, kw)
                )
                layer["biases"] = np.random.uniform(-1, 1, size=(layer["filters"]))

                current_height, current_width = output_height, output_width
                current_channels = layer["filters"]

            elif layer["type"] == "pool":
                # Adjust pool size if necessary
                pool_size = layer["size"]
                if pool_size > current_height:
                    pool_size = current_height
                if pool_size > current_width:
                    pool_size = current_width
                layer["size"] = pool_size

                # Recalculate output size
                output_height = compute_pool_output_size(current_height, pool_size, stride=layer.get("stride", 2))
                output_width = compute_pool_output_size(current_width, pool_size, stride=layer.get("stride", 2))
                current_height, current_width = output_height, output_width

        # Recalculate flattened size
        flattened_size = current_height * current_width * current_channels

        # Update FFN layers
        previous_size = flattened_size
        for layer in self.genome["ffn_layers"]:
            layer["weights"] = np.random.uniform(-1, 1, size=(layer["neurons"], previous_size))
            layer["biases"] = np.random.uniform(-1, 1, size=(layer["neurons"]))
            previous_size = layer["neurons"]



        
    
    
    
def compute_conv_output_size(input_size, kernel_size, stride=1, padding=0):
    return max(1, ((input_size - kernel_size + 2 * padding) // stride) + 1)


def compute_pool_output_size(input_size, pool_size, stride=2):
    return max(1, ((input_size - pool_size) // stride) + 1)

def fitness_wrapper(args):
    individual, shared_boards = args
    return individual.evaluate_fitness(shared_boards)

def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        genome = {"cnn_layers": [], "ffn_layers": []}
        # Randomly decide number of layers (e.g., 2 to 5)
        num_layers = random.randint(2, 5)
        for _ in range(num_layers):
            layer_type = random.choice(["conv", "pool"])
            if layer_type == "conv":
                filters = random.randint(4, 16)
                kernel_size = 3

                layer = {
                    "type": "conv",
                    "filters": filters,
                    "kernel_size": kernel_size,
                    "stride": 1,
                    "activation": random.choice(["relu", "tanh"]),
                    "weights": np.random.uniform(-1, 1, size=(filters, 1, kernel_size, kernel_size)),
                    "biases": np.random.uniform(-1, 1, size=(filters,)),
                }
            else:
                layer = {
                    "type": "pool",
                    "size": 2,
                    "stride": 2,
                    "pool_type": random.choice(["max", "avg"]),
                }
            genome["cnn_layers"].append(layer)
        # Define FFN layers
        genome["ffn_layers"] = [
            {"neurons": random.randint(16, 64), "activation": "relu", "weights": np.random.uniform(-1, 1, size=(random.randint(16, 64), 6*7*1)), "biases": np.random.uniform(-1, 1, size=(random.randint(16, 64),))},
            {"neurons": 7, "activation": "softmax", "weights": np.random.uniform(-1, 1, size=(7, random.randint(16, 64))), "biases": np.random.uniform(-1, 1, size=(7,))}
        ]
        # Create CNN
        cnn = CNN(genome=genome, input_size=(6, 7), input_channels=1)
        population.append(cnn)
    return population



def tournament_selection(winners, wins, tournament_size=3):
    selected = []
    for _ in range(len(winners)):
        participants = random.sample(list(zip(winners, wins)), tournament_size)
        winner = max(participants, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected


def nextGeneration(population, generation):
    global BEAT_SPAM

    # --- Shuffle population ---
    random.shuffle(population)

    # HEAD-TO-HEAD
    wins = [0 for _ in range(POPULATION_SIZE)]
    for i in range(POPULATION_SIZE):
        cnn1 = population[i]
        # Build model once if changed:

        for j in range(1, 3):
            cnn2 = population[(i + j) % POPULATION_SIZE]
            result = AI_duel(cnn1, cnn2)  # do the head-to-head
            if result == 1:
                wins[i] += 1
            elif result == -1:
                wins[(i + j) % POPULATION_SIZE] += 1
                
        beatSpam = 0
        for j in range(4):
            if learn_defense(cnn1):
                beatSpam += 1
                wins[i] += 1
            
        if beatSpam == 4 and BEAT_SPAM == False:
            population = [cnn1 for _ in range(POPULATION_SIZE)]
            BEAT_SPAM = True
            print("BEAT SPAM")
            play_game_with_ai(cnn1)
            break


    # SELECT TOP WINNERS
    top_winners_indices = heapq.nlargest(10, range(len(wins)), key=wins.__getitem__)
    winners = [population[idx] for idx in top_winners_indices]

    # Evaluate winners
    fitness_scores = [wins[idx] for idx in top_winners_indices]

    # PICK top 10 as elites
    next_gen = [cnn.copy() for cnn in winners]  # Use copies to avoid shared references

    # Offspring generation: fill the rest of the population
    num_to_generate = POPULATION_SIZE - len(next_gen)
    highest_fitness = max(fitness_scores)
    
    if highest_fitness == 8:
        play_game_with_ai(winners[0])

    for _ in range(num_to_generate):
        tournament_winner = tournament_selection(winners, fitness_scores, tournament_size=1)
        child = tournament_winner.copy().mutate(generation=generation)  # Pass generation number
        child.build_model()
        next_gen.append(child)


    # Ensure population size
    assert len(next_gen) == POPULATION_SIZE

    return next_gen, highest_fitness




def update_boards_dual(red_board, yellow_board, col, value):
    """
    Drop a piece for both the Red and Yellow boards simultaneously.
    Args:
        red_board (torch.Tensor): The board from Red's perspective.
        yellow_board (torch.Tensor): The board from Yellow's perspective.
        col (int): The column where the piece is dropped.
        value (int): The value of the piece (1 for Red, -1 for Yellow).
    """
    for row in range(5, -1, -1):  # Find the first empty row from the bottom
        if red_board[0, 0, row, col] == 0:
            red_board[0, 0, row, col] = value
            yellow_board[0, 0, row, col] = -value  # Update Yellow's board
            return


def AI_duel(cnn1, cnn2):
    # Initialize boards
    yellow_board = create_board()
    red_board = create_board()
    current_value = -1

    # Simulate a game between cnn1 and cnn2
    while True:
        if current_value == 1:
            # CNN1 (Yellow) plays using the Red board
            output, _ = cnn1.forward_pass(yellow_board)
            predicted_col = select_best_available_column(output[0], yellow_board[0, 0])
        else:
            # CNN2 (Red) plays using the Yellow board
            output, _ = cnn2.forward_pass(red_board)
            predicted_col = select_best_available_column(output[0], red_board[0, 0])

        # Update both boards with the move
        last_row = None
        if current_value == 1:
            last_row = drop_piece(yellow_board[0, 0], predicted_col, 1)
            drop_piece(red_board[0, 0], predicted_col, 1)
        else:
            last_row = drop_piece(red_board[0, 0], predicted_col, -1)
            drop_piece(yellow_board[0, 0], predicted_col, -1)

        # Check for a winner
        if check_win(yellow_board[0, 0] if current_value == -1 else red_board[0, 0], last_row, predicted_col, current_value):
            if current_value == 1:
                return 1
            else:
                return -1

        # Check for a draw
        if torch.all(yellow_board[0, 0] != 0):  # Board is full
            return 0

        current_value *= -1  # Switch turns
        
        
def learn_defense(cnn1):
    """
    Simulates a game where the CNN plays against a bot that always picks the same column
    until it is full. Returns True for a win or draw for the CNN.
    Args:
        cnn1 (CNN): The CNN model to play as Yellow.
    Returns:
        bool: True if CNN wins or draws, False otherwise.
    """
    # Initialize the board
    board = create_board()
    current_value = -1  # Person (Red) starts
    offenseColumn = random.randint(0, 6)  # Opponent's preferred column
    lastRow = 0
    lastCol = 0

    while True:
        if current_value == 1:
            # CNN1 (Yellow - AI) plays
            output, _ = cnn1.forward_pass(board)
            cnnColumn = select_best_available_column(output[0], board[0, 0])
            lastRow = drop_piece(board[0, 0], cnnColumn, 1)  # Drop AI's piece
            lastCol = cnnColumn
        else:
            # Opponent (Red) plays
            validRow = drop_piece(board[0, 0], offenseColumn, -1)  # Try to drop in offenseColumn
            while validRow == -1:  # If column is full, pick a new random column
                offenseColumn = random.randint(0, 6)
                validRow = drop_piece(board[0, 0], offenseColumn, -1)
            lastRow = validRow
            lastCol = offenseColumn

        # Check for a winner
        if check_win(board[0, 0], lastRow, lastCol, current_value):
            return current_value == 1  # True if CNN (Yellow) wins, False otherwise

        # Check for a draw
        if torch.all(board[0, 0] != 0):  # Board is full
            return True

        # Switch turns
        current_value *= -1


def play_game_with_ai(ai_model):
    """
    Play a game of Connect Four between the most fit AI and the player.
    The AI plays as Yellow (1), and the player plays as Red (-1).

    Args:
        ai_model (CNN): The most fit CNN model.
    """
    board = create_board()
    current_value = -1  # 1 for Yellow (AI), -1 for Red (Player)
    moves_made = 0

    while True:
        print_board(board[0, 0])  # Print the board in a human-readable format
        if current_value == 1:  # AI's turn
            output, _ = ai_model.forward_pass(board)  # AI predicts column scores
            predicted_col = select_best_available_column(output[0], board[0, 0])

            for row in range(5, -1, -1):  # Drop the piece in the predicted column
                if board[0, 0, row, predicted_col] == 0:
                    board[0, 0, row, predicted_col] = 1
                    break

            print(f"\nAI (Yellow) chooses column {predicted_col + 1}")
        else:  # Player's turn
            while True:
                try:
                    col = int(input("\nYour turn (Red), choose a column (1-7): ")) - 1
                    if is_valid_move(board[0, 0], col):
                        for row in range(5, -1, -1):
                            if board[0, 0, row, col] == 0:
                                board[0, 0, row, col] = -1
                                break
                        break
                    print("Invalid move. Column is either full or out of range.")
                except ValueError:
                    print("Please enter a valid number between 1 and 7.")

        moves_made += 1

        # Check for a winner
        for r in range(6):
            for c in range(7):
                if board[0, 0, r, c] != 0 and check_win(board[0, 0], r, c, board[0, 0, r, c]):
                    print_board(board[0, 0])
                    if board[0, 0, r, c] == 1:
                        print("\nAI (Yellow) wins!")
                    else:
                        print("\nYou (Red) win!")
                    return

        # Check for a draw
        if moves_made == 42:
            print_board(board[0, 0])
            print("\nIt's a draw!")
            return

        # Switch turns
        current_value *= -1




import time

if __name__ == '__main__':
    print("Creating Population")
    # Initialize population with diverse architectures
    population = initialize_population()
    print("Beginning Neuroevolution")

    # List to store fitness values
    fitness_history = []
    # List to store generation times
    time_history = []
    generation = 0

    while True:
        start_time = time.time()  # Start timing the generation

        # Generate the next generation
        population, highest_fitness = nextGeneration(population, generation)

        # Record generation time
        generation_time = time.time() - start_time
        time_history.append(generation_time)

        # Add the highest fitness of this generation to history
        fitness_history.append(highest_fitness)

        print(f"Generation {generation + 1}, Highest Fitness: {highest_fitness:.2f}, Time: {generation_time:.2f} seconds")

        # Every 10 generations, compute and print the average fitness and average time
        if (generation + 1) % 10 == 0:
            avg_fitness = sum(fitness_history[-10:]) / 10
            avg_time = sum(time_history[-10:]) / 10
            print(f"Average Fitness over the last 10 generations: {avg_fitness:.2f}")
            print(f"Average Time per Generation over the last 10 generations: {avg_time:.2f} seconds")

        # Every 50 generations, play a game with the most fit AI and print its architecture
        if (generation) % 500 == 0:
            print(f"\nGeneration {generation + 1}: Playing a game with the most fit AI.")
            play_game_with_ai(population[0])  # Play with the best AI model
            print("\nArchitecture of the most fit AI after this generation:")
            population[0].save_architecture_to_file(f"generation_{generation}_cnn_architecture.json")
            
        generation += 1