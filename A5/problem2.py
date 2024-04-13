import torch.nn as nn

class CNN(nn.Module):
    """
    Constructs a convolutional neural network according to the architecture in the exercise sheet using the layers in torch.nn.

    Args:
        num_classes: Integer stating the number of classes to be classified.
    """

    def __init__(self, num_classes=2):
        super(CNN, self).__init__()

        # Assuming a simplified structure where 'conv' represents a single convolutional operation
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers sequence
        self.fc = nn.Sequential(
            nn.Linear(4*4*256, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim = 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class LossMeter(object):
    """
    Constructs a loss running meter containing the methods reset, update and get_score. 
    Reset sets the loss and step to 0. Update adds up the loss of the current batch and updates the step.
    get_score returns the runnung loss.

    """

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss = 0.
        self.step = 0.

    def update(self, loss):
        self.loss += loss
        self.step += 1

    def get_score(self):
        return self.loss / self.step


def analysis():
    """
    Compare the performance of the two networks in Problem 1 and Problem 2 and
    briefly summarize which one performs better and why.
    """

    print("""
        Analysis for models trained in problem01 and problem02 using the TEST SET:
          
            - The accuracy of the model trained in problme01 with the default parameters is approx. 0.56 (56%).
            - The accuracy of the model trained in problme01 reaches 1.0 (100%) if the following parameters 
              are set as: lr = 8e-4, batch_size = 32, num_epochs = 400.

            - The accuracy of the model trained in problem02 is approx. 0.54 (54%) in the epochs 1 to 4, in epoch 5
              the accuracy is approx. 0.6 (60%). The 6th epoch has a sudden increase in the accuracy to 0.9375 (93.75%). 
              In epoch 7, the model reaches an accuracy of 1.0 (100%). 

            - Conclusion: The model trained in problem02 has better accuracy than the model trained in problem01. 
              The reason for the difference in performance is that model in problem01 does not have convolution layers. 
              Convolution layers help in learning the semantic information in the image. For image data, there are 
              spatial features in the image which can be learned with convolution layers.
          """)
