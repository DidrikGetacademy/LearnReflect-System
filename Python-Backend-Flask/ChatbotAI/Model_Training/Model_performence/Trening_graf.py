import matplotlib.pyplot as plt
import os


training_losses_object = []
eval_loss_object = []
epochs = []


def read_Trainingdata():
    global training_losses_object, eval_loss_object, epochs  

    # Path to the file
    file_path = os.path.join(r'C:\Users\Didrik\OneDrive\Skrivebord\kopi\LearnReflect Project\LearnReflect Project\Python-Backend-Flask\ChatbotAI\Model_Training\Training_data\Diagram_eval_trainloss.txt')

    if os.path.exists(file_path):
        print(f"File found: {file_path}")
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                if "Loss:" in line:
                    # Use square brackets to create a list of float values
                    loss_values = [float(val.strip()) for val in line.split(':')[1].split(',')]
                    training_losses_object.extend(loss_values)
                    print(f"Training Losses: {training_losses_object}")
                    
                elif "Eval-loss:" in line:
                    # Similarly for eval loss
                    eval_loss_values = [float(val.strip()) for val in line.split(':')[1].split(',')]
                    eval_loss_object.extend(eval_loss_values)
                    print(f"Eval Losses: {eval_loss_object}")
                    
                elif "epochs" in line:
                    # Similarly for epochs
                    epochs_value = [int(val.strip()) for val in line.split(':')[1].split(',')]
                    epochs.extend(epochs_value)
                    print(f"Epochs: {epochs}")
        
    else:
        print(f"File not found: {file_path}")
    
    return training_losses_object, eval_loss_object, epochs


training_losses_object, eval_loss_object, epochs = read_Trainingdata()


def Training_Evolve_Diagram(epochs_return, eval_losses_return, training_losses_return):
    plt.plot(epochs_return, training_losses_return, label='Trenings Tap', marker='o', linestyle='--', color='blue')
    plt.plot(epochs_return, eval_losses_return, label='Evaluerings Tap', marker='x', linestyle='-', color='red')
    
    if any(eval_loss > train_loss for eval_loss, train_loss in zip(eval_losses_return, training_losses_return)):
         print("Advarsel: Evaluerings Tapet har overskredet Trenings Tapet i løpet av treningen!")
         
         
    plt.title('Trenings og Evaluerings Tap over Epoker')
    plt.xlabel('Epoker')
    plt.ylabel('Tap')
    plt.legend()
    plt.grid()
    
    
    
    plt.show()


#Training_Evolve_Diagram(epochs, eval_loss_object, training_losses_object)

