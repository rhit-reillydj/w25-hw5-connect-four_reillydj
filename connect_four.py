import copy
import json
from multiprocessing import Pool
import time

import random
import numpy as np
import heapq

import torch
import torch.nn as nn


POPULATION_SIZE = 50
BATCH_SIZE = 5

KERNEL_SIZE = 3
CONV_STRIDE = 1
POOL_SIZE = 2
POOL_STRIDE = 2
CHANNELS = 8

def create_board():
    """
    Create an empty Connect Four board as a PyTorch tensor, padded with -2 around the edges.
    
    Returns:
        torch.Tensor: A 2D tensor of shape (8, 9), where the playable area (6x7) is initialized with zeros,
                      and the border is padded with -2.
    """
    # Create a 6x7 board filled with zeros
    board = torch.zeros((6, 7), dtype=torch.float32)

    # Pad the board with -2 on all sides
    padded_board = torch.full((8, 9), -2, dtype=torch.float32)
    
    # Place the 6x7 board inside the padded area
    padded_board[1:7, 1:8] = board

    return padded_board


def print_board(board):
    """
    Print the Connect Four board in a human-readable format.
    
    Args:
        board (torch.Tensor): A 2D tensor representing the game board.
    """
    # Map numbers to symbols for display
    symbol_map = {0: ' ', 1: 'Y', -1: 'R', -2: 'O'}
    print('\n X 1 2 3 4 5 6 7 X')
    print('------------------')
    for row in board:
        print('|' + '|'.join(symbol_map[cell.item()] for cell in row) + '|')
    print('------------------')


def is_valid_move(board, col):
    """
    Check if a move is valid (column is not full).
    
    Args:
        board (torch.Tensor): A 2D tensor representing the game board.
        col (int): The internal column index (0-6).
    
    Returns:
        bool: True if the move is valid, False otherwise.
    """
    if 1 <= col <= 7:
        return board[1, col] == 0  # +1 to account for padding
    return False


def drop_piece(board, col, value):
    """
    Drop a piece into the board in the specified column.

    Args:
        board (torch.Tensor): A 2D tensor representing the game board.
        col (int): The internal column index (0-6).
        value (int): The value of the piece to drop (1 for Red, -1 for Yellow).

    Returns:
        tuple: (row index where the piece was dropped, internal column index),
               or (-1, -1) if the column is full.
    """
    for row in range(6, -1, -1):
        if board[row, col] == 0:  # +1 for padding
            board[row, col] = value
            return row, col
    return -1, -1



def check_win(board, last_row, last_col, value):
    """
    Check if the last move results in a win for the current player.

    Args:
        board (torch.Tensor): A 2D tensor representing the game board.
        last_row (int): The row index of the last move.
        last_col (int): The column index of the last move.
        value (int): The value of the current player (1 for Red, -1 for Yellow).

    Returns:
        bool: True if the player wins, False otherwise.
    """
    
    # Ensure valid move was made
    if last_row == -1 or last_col == -1:
        return False  

    # Define four possible winning directions
    directions = [
        [(0, 1), (0, -1)],  # Horizontal
        [(1, 0), (-1, 0)],  # Vertical
        [(1, 1), (-1, -1)], # Diagonal positive slope
        [(1, -1), (-1, 1)]  # Diagonal negative slope
    ]
    
    def count_direction(row, col, row_step, col_step):
        count = 0
        r, c = row + row_step, col + col_step  # Start from next position
        while 1 <= r <= 6 and 1 <= c <= 7 and board[r, c] == value:
            count += 1
            r += row_step
            c += col_step
        return count
    
    for direction_pair in directions:
        total_count = 1  # Include the last move
        for row_step, col_step in direction_pair:
            total_count += count_direction(last_row, last_col, row_step, col_step)
            if total_count >= 4:
                return True  # Winning condition met
    
    return False  # No win detected




def play_game():
    """
    Play a game of Connect Four with the board represented as a PyTorch tensor.
    """
    board = create_board()
    current_value = -1  # 1 for Red, -1 for Yellow
    moves_made = 0
    
    while True:
        print_board(board)
        player_name = "Red" if current_value == -1 else "Yellow"
        
        while True:
            try:
                col = int(input(f"\n{player_name}'s turn (1-7): "))
                if is_valid_move(board, col):
                    break
                print("Invalid move. Column is either full or out of range.")
            except ValueError:
                print("Please enter a number between 1 and 7.")
        
        row, col = drop_piece(board, col, current_value)
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

        
        
        
def AI_duel(cnn1, cnn2):
    # Initialize boards
    yellow_board = create_board()
    red_board = create_board()
    current_value = -1

    # Simulate a game between cnn1 and cnn2
    while True:
        if current_value == 1:
            # CNN1 (Yellow) plays using the Red board
            predicted_col = cnn1.forward_pass(yellow_board)
        else:
            # CNN2 (Red) plays using the Yellow board
            predicted_col = cnn2.forward_pass(red_board)

        # Update both boards with the move
        last_row = None
        if current_value == 1:
            last_row, _ = drop_piece(yellow_board, predicted_col, current_value)
            drop_piece(red_board, predicted_col, -current_value)
            
        else:
            last_row, _ = drop_piece(red_board, predicted_col, current_value)
            drop_piece(yellow_board, predicted_col, -current_value)

        # Check for a winner
        if check_win(yellow_board if current_value == 1 else red_board, last_row, predicted_col, current_value):
            if current_value == 1:
                return 1
            else:
                return -1

        # Check for a draw
        if torch.all(yellow_board != 0):  # Board is full
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
    current_value = -1  # Red starts
    offenseColumn = random.randint(1, 7)  # Opponent's preferred column (internal indexing 0-6)

    while True:
        if current_value == 1:
            # CNN1 (Yellow - AI) plays
            cnnColumn = cnn1.forward_pass(board)
            lastRow, lastCol = drop_piece(board, cnnColumn, 1)  # Drop AI's piece
            
        else:
            # Opponent (Red) plays
            validRow, _ = drop_piece(board, offenseColumn, -1)  # Try to drop in offenseColumn

            # If the column is full, move to the next available column
            if validRow == -1:
                originalColumn = offenseColumn
                while True:
                    offenseColumn = offenseColumn + 1 if offenseColumn < 7 else 1
                    validRow, _ = drop_piece(board, offenseColumn, -1)
                    if validRow != -1 or offenseColumn == originalColumn:
                        break  # Stop if a move is made or if all columns are full

            # Update last move position
            lastRow, lastCol = validRow, offenseColumn

        # Check for a winner
        if lastRow != -1 and check_win(board, lastRow, lastCol, current_value):
            return current_value == 1  # True if CNN (Yellow) wins, False otherwise

        # Check for a draw
        if torch.all(board != 0):  # Board is full
            return True

        # Switch turns
        current_value *= -1




        
        
        
        
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




def select_best_available_column(output_scores, board):
    """
    Select the best available column from the output scores.

    Args:
        output_scores (torch.Tensor): The output scores for each column.
        board (torch.Tensor): The current board.

    Returns:
        int: The internal best available column index (0-6).
    """
    sorted_columns = torch.argsort(output_scores, descending=True)
    for col in sorted_columns:
        col = col.item() + 1
        if is_valid_move(board, col):
            return col
    return -1  # This should not happen unless all columns are full



    
























# AI Part
class GenomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # We will dynamically create modules once we know the genome.
        self.cnn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

    def load_from_genome(self, genome):
        """
        Clear existing modules and build new ones according to genome specs.
        Then load the weights/biases from the genome into these modules.
        """
        self.cnn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()

        current_in_channels = 1  # We expect (batch_size, 1, 6, 7) input originally

        # Build CNN
        for layer in genome["cnn_layers"]:
            if layer["type"] == "conv":
                conv = nn.Conv2d(
                    in_channels=current_in_channels,
                    out_channels=layer["filters"],
                    kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
                    stride=layer.get("stride", 1),
                    bias=True
                )
                # Load weights
                w = torch.tensor(layer["weights"], dtype=torch.float32)
                b = torch.tensor(layer["biases"], dtype=torch.float32)
                with torch.no_grad():
                    conv.weight.copy_(w)
                    conv.bias.copy_(b)

                self.cnn_layers.append(conv)
                current_in_channels = layer["filters"]

            elif layer["type"] == "pool":
                if layer["pool_type"] == "max":
                    pool = nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_STRIDE)
                else:
                    pool = nn.AvgPool2d(kernel_size=POOL_SIZE, stride=POOL_SIZE)
                self.cnn_layers.append(pool)

        # Initialize FFN layers with correct in_features
        current_in_features = 64  # Matches the flattened CNN output
        for layer in genome["ffn_layers"]:
            dense = nn.Linear(in_features=current_in_features, out_features=layer["neurons"], bias=True)

            # Load weights and biases
            w = torch.tensor(layer["weights"], dtype=torch.float32)
            b = torch.tensor(layer["biases"], dtype=torch.float32)
            with torch.no_grad():
                # Ensure weight shapes match
                expected_weight_shape = (layer["neurons"], current_in_features)
                if w.shape != expected_weight_shape:
                    raise ValueError(f"FFN layer weights shape mismatch. Expected {expected_weight_shape}, got {w.shape}")
                dense.weight.copy_(w)
                dense.bias.copy_(b)

            self.ffn_layers.append(dense)
            current_in_features = layer["neurons"]

    def forward(self, x):
        # CNN forward
        for mod in self.cnn_layers:
            x = mod(x)  # either a conv or a pool
            if isinstance(mod, nn.Conv2d):
                # Handle activations if stored in genome (not implemented here)
                pass

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # FFN forward
        for layer in self.ffn_layers:
            x = layer(x)  # linear
        return x












class CNN:
    def __init__(self, genome=None):
        """
        Initialize the CNN with a genome. If no genome is provided, a random one is created.

        Args:
            genome (dict, optional): Predefined genome structure. Defaults to None.
            input_size (tuple): Input dimensions (height, width). Defaults to (6, 7).
            input_channels (int): Number of input channels. Defaults to 1.
        """
        if genome:
            self.genome = genome
        else:
            self.genome = {"cnn_layers": [], "ffn_layers": []}

            # Define CNN layers
            cnn_layers = [
                {"type": "conv", "filters": CHANNELS, "kernel_size": KERNEL_SIZE, "stride": CONV_STRIDE, "activation": "relu"},
                {"type": "pool", "size": POOL_SIZE, "stride": POOL_STRIDE, "pool_type": "max"},
                {"type": "conv", "filters": CHANNELS * CHANNELS, "kernel_size": KERNEL_SIZE, "stride": CONV_STRIDE, "activation": "relu"},
            ]

            current_in_channels = 1  # Starting with 1 input channel
            # Process CNN layers
            for layer in cnn_layers:
                if layer["type"] == "conv":

                    layer["weights"] = np.random.uniform(
                        -1, 1, size=(layer["filters"], current_in_channels, layer["kernel_size"], layer["kernel_size"])
                    )
                    layer["biases"] = np.random.uniform(-1, 1, size=(layer["filters"]))
                    current_in_channels = layer["filters"]  # Update for next layer

                self.genome["cnn_layers"].append(layer)

            # Define FFN layers
            ffn_layers = [
                {"neurons": CHANNELS * CHANNELS, "activation": "relu"},      # Input
                {"neurons": 7, "activation": "softmax"},    # Output
            ]

            previous_size = 64      # Set to 64 to match CNN's flattened output
            for layer in ffn_layers:
                layer["weights"] = np.random.uniform(-1, 1, size=(layer["neurons"], previous_size))
                layer["biases"] = np.random.uniform(-1, 1, size=(layer["neurons"]))
                previous_size = layer["neurons"]
                self.genome["ffn_layers"].append(layer)

        # Initialize the model that will be built dynamically from the genome
        self.model = GenomeModel()
    
    def build_model(self):
        self.model.load_from_genome(self.genome)
            
    def copy(self):
        # Use deepcopy to ensure the entire object is copied without shared references
        return copy.deepcopy(self)

                
    @classmethod
    def initialize_from_architecture(cls, architecture):
        """
        Initialize a CNN using the printed architecture.
        
        Args:
            architecture (dict): The architecture dictionary printed by `print_architecture`.
        
        Returns:
            CNN: A new CNN instance initialized with the specified architecture.
        """
        return cls(genome=architecture)
    
    def save_architecture_to_file(self, file_path="cnn_architecture.json"):
        """
        Save the architecture of the CNN to a JSON file for later use.
        
        Args:
            file_path (str): The path to the file where the architecture will be saved.
        """
        def serialize_layer(layer):
            """
            Helper function to convert a layer's weights and biases into JSON-serializable formats.
            """
            serialized_layer = layer.copy()

            if "weights" in serialized_layer and isinstance(serialized_layer["weights"], np.ndarray):
                serialized_layer["weights"] = serialized_layer["weights"].tolist()
            elif "weights" in serialized_layer:
                serialized_layer["weights"] = None

            if "biases" in serialized_layer and isinstance(serialized_layer["biases"], np.ndarray):
                serialized_layer["biases"] = serialized_layer["biases"].tolist()
            elif "biases" in serialized_layer:
                serialized_layer["biases"] = None

            return serialized_layer

        try:
            architecture = {
                "cnn_layers": [serialize_layer(layer) for layer in self.genome.get("cnn_layers", [])],
                "ffn_layers": [serialize_layer(layer) for layer in self.genome.get("ffn_layers", [])]
            }

            with open(file_path, "w") as file:
                json.dump(architecture, file, indent=4)
            print(f"Architecture saved to {file_path}")

        except Exception as e:
            print(f"Error while saving architecture: {e}")

    def forward_pass(self, input_tensor):
        """
        Actually do the forward pass with the model we built.
        NOTE: We assume build_model() was called if the genome changed.
        """
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
        # The final layer shape might be (batch_size, 7)
        # Convert to predicted columns, etc.
        board_2d = input_tensor.squeeze(0).squeeze(0)  # [8, 9]

        predicted_column = select_best_available_column(output[0], board_2d)
        return predicted_column








    
    def mutate(self, generation=0):
        """
        Apply mutations with an adaptive rate based on the generation number.
        
        Args:
            generation (int): Current generation number.
        """
        # Base mutation weights
        base_mutation_weights = {
            "modify_layer_params": 0.40,
            "alter_weights_biases": 0.40,
            "add_ffn_neurons": 0.03,
            "add_ffn_layer": 0.01,
            "remove_ffn_neurons": 0.02,
            "remove_ffn_layer": 0.01
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
            if mutation_type == "modify_layer_params":
                self.modify_layer_params()
            elif mutation_type == "alter_weights_biases":
                self.alter_weights_biases()
            elif mutation_type == "add_ffn_neurons":
                self.add_ffn_neurons()
            elif mutation_type == "add_ffn_layer":
                self.add_ffn_layer()
            elif mutation_type == "remove_ffn_neurons":
                self.remove_ffn_neurons()
            elif mutation_type == "remove_ffn_layer":
                self.remove_ffn_layer()
                
        return self


    def modify_layer_params(self, param_change_prob=0.5, weight_change_std=0.1):
        """
        Modify the parameters of a random layer in the genome.
        Each parameter has an independent probability of being modified.
        Weights and biases are perturbed rather than completely replaced.
        
        Args:
            param_change_prob (float): Probability of changing each parameter (0.0 to 1.0).
            weight_change_std (float): Standard deviation of the noise added to weights and biases.
        """
        layer = random.choice(self.genome["cnn_layers"])
        
        if layer["type"] == "conv":
            # Randomly modify parameters with probability
            if random.random() < param_change_prob:
                layer["activation"] = random.choice(["relu", "sigmoid", "tanh"])

            # Perturb weights and biases rather than replacing them
            if "weights" in layer and layer["weights"] is not None:
                noise = np.random.normal(0, weight_change_std, layer["weights"].shape)
                layer["weights"] += noise
            
            if "biases" in layer and layer["biases"] is not None:
                noise = np.random.normal(0, weight_change_std, layer["biases"].shape)
                layer["biases"] += noise
        
        elif layer["type"] == "pool":
            if random.random() < param_change_prob:
                layer["pool_type"] = random.choice(["max", "avg"])


    def alter_weights_biases(self):
        """
        Apply small random perturbations to weights and biases
        """
        noise_scale = 0.1
        for layer in self.genome["cnn_layers"]:
            if "weights" in layer:
                layer["weights"] += np.random.uniform(-noise_scale, noise_scale, size=layer["weights"].shape)
            if "biases" in layer:
                layer["biases"] += np.random.uniform(-noise_scale, noise_scale, size=layer["biases"].shape)
                    
        for layer in self.genome["ffn_layers"]:
            layer["weights"] += np.random.uniform(-noise_scale, noise_scale, size=layer["weights"].shape)
            layer["biases"] += np.random.uniform(-noise_scale, noise_scale, size=layer["biases"].shape)


    def add_ffn_neurons(self, max_neurons=128):
        """
        Add neurons to a random hidden fully connected layer, ensuring the total number
        of neurons in the layer does not exceed `max_neurons`.
        
        Args:
            max_neurons (int): The maximum allowed number of neurons in a layer.
        """
        # Ensure there are hidden layers to modify
        if len(self.genome["ffn_layers"]) > 2:  # Exclude the input and output layers
            # Randomly choose a hidden layer (excluding input and output layer)
            layer_index = random.randint(1, len(self.genome["ffn_layers"]) - 2)
            layer = self.genome["ffn_layers"][layer_index]
            
            # Calculate how many neurons can be added without exceeding the limit
            current_neurons = layer["neurons"]
            
            max_addable_neurons = max_neurons - current_neurons

            if max_addable_neurons > 0:  # Only proceed if there’s room for more neurons
                # Add neurons
                neurons_to_add = random.randint(1, min(16, max_addable_neurons))
                new_weights = np.random.uniform(-1, 1, size=(neurons_to_add, layer["weights"].shape[1]))
                new_biases = np.random.uniform(-1, 1, size=(neurons_to_add))
                layer["weights"] = np.vstack([layer["weights"], new_weights])
                layer["biases"] = np.concatenate([layer["biases"], new_biases])
                layer["neurons"] += neurons_to_add
                
    def add_ffn_layer(self, max_hidden_layers=3):
        """
        Add neurons to a random hidden fully connected layer, ensuring the total number
        of neurons in the layer does not exceed `max_neurons`.
        
        Args:
            max_neurons (int): The maximum allowed number of neurons in a layer.
        """
        # Ensure there are hidden layers to modify
        length = len(self.genome["ffn_layers"])
        if length <= max_hidden_layers + 2 and length > 2:  # Ensure no more than 3 hidden layers
            
            layer_index = random.randint(1, length - 2)

            # Define FFN layers
            numNeuronsPower = random.randint(4, 7)
            numNeurons = 2 ** numNeuronsPower
            activation = random.choice(["relu", "sigmoid", "tanh"])
            new_layer = {"neurons": numNeurons, "activation": activation}
            previous_size = self.genome["ffn_layers"][layer_index - 1]["neurons"]

            new_layer["weights"] = np.random.uniform(-1, 1, size=(new_layer["neurons"], previous_size))
            new_layer["biases"] = np.random.uniform(-1, 1, size=(new_layer["neurons"]))

            previous_size = new_layer["neurons"]
                
            # Randomly choose a hidden layer (excluding input and output layer)
            self.genome["ffn_layers"].insert(layer_index, new_layer)



    def remove_ffn_neurons(self):
        """Remove neurons from a random hidden fully connected layer."""
        # Ensure there are hidden layers to modify
        if len(self.genome["ffn_layers"]) > 2:  # Exclude the input and output layer
            # Randomly choose a hidden layer (excluding input and output layer)
            layer_index = random.randint(1, len(self.genome["ffn_layers"]) - 2)
            layer = self.genome["ffn_layers"][layer_index]
            
            # Remove neurons
            if layer["neurons"] > 1:  # Ensure at least one neuron remains
                neurons_to_remove = random.randint(1, min(16, layer["neurons"] - 1))
                layer["weights"] = layer["weights"][:-neurons_to_remove, :]
                layer["biases"] = layer["biases"][:-neurons_to_remove]
                layer["neurons"] -= neurons_to_remove
                
                
    def remove_ffn_layer(self):
        """
        Add neurons to a random hidden fully connected layer, ensuring the total number
        of neurons in the layer does not exceed `max_neurons`.
        
        Args:
            max_neurons (int): The maximum allowed number of neurons in a layer.
        """
        # Ensure there are hidden layers to modify
        if len(self.genome["ffn_layers"]) > 2:  # Ensure at least 1 hidden layer
            
            layer_index = random.randint(1, len(self.genome["ffn_layers"]) - 2)
                
            # Randomly choose a hidden layer (excluding input and output layer)
            self.genome["ffn_layers"].remove(layer_index)




def controlled_layer_crossover(parent1, parent2):
    """
    Perform controlled layer selection crossover between two parents, ensuring
    every output matches its associated input by adjusting/truncating weights from parents.
    
    Args:
        parent1 (CNN): The first parent CNN.
        parent2 (CNN): The second parent CNN.
        input_size (tuple): The (height, width) of the CNN input.
        max_layers (int): Max number of layers in the child; if None, use max from parents.
    
    Returns:
        CNN: A child CNN with dimension/channel-consistent layers taken from both parents.
    """

    # Process CNN layers
    cnn_layers = []
    for layerIndex in range(3):
        parent = parent1 if random.random() < 0.5 else parent2
        cnn_layers.append(parent.genome["cnn_layers"][layerIndex])
        
    ffn_layers = []
    parent1FFNLayers = len(parent1.genome["ffn_layers"])
    parent2FFNLayers = len(parent2.genome["ffn_layers"])
    
    parent1Smaller = parent1FFNLayers < parent2FFNLayers

    for layerIndex in range((parent1FFNLayers - 1) if parent1Smaller else (parent2FFNLayers - 1)):
        parent = parent1 if random.random() < 0.5 else parent2
        ffn_layers.append(parent.genome["ffn_layers"][layerIndex])
        
    if parent1Smaller:
        ffn_layers.append(parent1.genome["ffn_layers"][parent1FFNLayers - 1])
    else:
        ffn_layers.append(parent2.genome["ffn_layers"][parent2FFNLayers - 1])
        
    child_genome = {"cnn_layers": cnn_layers, "ffn_layers": ffn_layers}
    child_cnn = CNN.initialize_from_architecture(child_genome)
        
    return child_cnn.copy()



def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        # Create CNN
        cnn = CNN()
        cnn.build_model()
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
                
        for j in range(4):
            if learn_defense(cnn1):
                wins[i] += 1


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

    for _ in range(num_to_generate):
        tournament_winners = tournament_selection(winners, fitness_scores, tournament_size=2)
        parent1 = tournament_winners[0]
        parent2 = tournament_winners[1]
        child = controlled_layer_crossover(parent1, parent2)
        child = child.mutate(generation=generation)  # Pass generation number
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
    last_move_row = -1
    last_move_col = -1

    while True:
        print_board(board)  # Print the board in a human-readable format
        if current_value == 1:  # AI's turn
            predicted_col = ai_model.forward_pass(board)  # AI predicts column scores
            
            if predicted_col == -1:
                print("\nAI cannot make a move. It's a draw!")
                return

            last_move_row, last_move_col = drop_piece(board, predicted_col, current_value)

            print(f"\nAI (Yellow) chooses column {predicted_col}")
        else:  # Player's turn
            while True:
                try:
                    col = int(input("\nYour turn (Red), choose a column (1-7): "))
                    if is_valid_move(board, col):
                        last_move_row, last_move_col = drop_piece(board, col, current_value)
                        break
                    else:
                        print("Invalid move. Column is either full or out of range.")
                except ValueError:
                    print("Please enter a valid number between 1 and 7.")

        moves_made += 1

        # Check for a winner only at the last move
        if last_move_row != -1 and check_win(board, last_move_row, last_move_col, board[last_move_row, last_move_col]):
            print_board(board)
            winner = "AI (Yellow)" if board[last_move_row, last_move_col] == 1 else "You (Red)"
            print(f"\n{winner} wins!")
            return

        # Check for a draw
        if moves_made == 42:
            print_board(board)
            print("\nIt's a draw!")
            return

        # Switch turns
        current_value *= -1







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
        if (generation) % 250 == 0:
            print(f"\nGeneration {generation + 1}: Playing a game with the most fit AI.")
            play_game_with_ai(population[0])  # Play with the best AI model
            print("\nArchitecture of the most fit AI after this generation:")
            population[0].save_architecture_to_file(f"generation_{generation}_cnn_architecture.json")
            
        generation += 1