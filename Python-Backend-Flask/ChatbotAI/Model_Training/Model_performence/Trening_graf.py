import matplotlib.pyplot as plt
import os

# Initialize the lists to store training loss, evaluation loss, and epoch data
training_losses_object = []
eval_loss_object = []
epochs = []


def read_Trainingdata():

    
    # Path to the file
    file_path = os.path.join(r'C:\Program Files (x86)\LearnReflect Project\LearnReflect Project\Python-Backend-Flask\ChatbotAI\Model_Training\Training_data_diagram\Diagram_eval_trainloss.txt')

    if os.path.exists(file_path):
        print(f"File found: {file_path}")
        
        with open(file_path, 'r') as file:
            
            lines = file.readlines()

            if len(lines) > 0:
                training_losses_object = [float(x) for x in lines[0].strip().split(',')]
                print(f"Training Losses: {training_losses_object}")
                
            if len(lines) > 1:
 
                eval_loss_object = float(lines[1].strip())
                print(f"Evaluation Loss: {eval_loss_object}")
                
            if len(lines) > 2:

                epochs = int(lines[2].strip())
                print(f"Epochs: {epochs}")
    else:
        print(f"File not found: {file_path}")
    
    return training_losses_object, eval_loss_object, epochs


training_losses_object, eval_loss_object, epochs = read_Trainingdata()

            
def Training_Evolve_Diagram(epochs_return, eval_losses_return, training_losses_return):
  
    plt.plot(epochs_return, training_losses_return, label='Trenings Tap', marker='o', linestyle='-', color='blue')
    plt.plot(epochs_return, eval_losses_return, label='Evaluerings Tap', marker='o', linestyle='-', color='red')

    for i in range(1, len(eval_losses_return) - 1):
        if eval_losses_return[i + 1] > eval_losses_return[i - 1]:
            plt.axvline(x=epochs_return[i], color='purple', linestyle=':', label=f'Overfitting starter ved Epoke {epochs_return[i]}')
            plt.text(epochs_return[i], eval_losses_return[i], f'Overfitting starter ved Epoke {epochs_return[i]}', color='purple', fontsize=10)
            break

    plt.xlabel('Epoker')
    plt.ylabel('Tap')
    plt.title('Trenings og Evaluerings Tap over Epoker')
    plt.legend()
    plt.grid()
    plt.show()


# Call the diagram function to execute
Training_Evolve_Diagram(epochs, eval_loss_object,training_losses_object)
