import torch


class OCRNet(torch.nn.Module):
    height = 128
    width = 64
    channels = 1
    conv_filters = 16
    conv_kernel_size = (3, 3)
    conv_padding = 1
    pool_size = 2
    linear_size = 32
    rnn_size = 512

    def __init__(self, num_classes=23):
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = torch.nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.conv_filters,
            kernel_size=self.conv_kernel_size,
            padding=self.conv_padding,
        )
        self.activation1 = torch.nn.ReLU()
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=self.pool_size)

        self.conv2 = torch.nn.Conv2d(
            in_channels=self.conv_filters,
            out_channels=self.conv_filters,
            kernel_size=self.conv_kernel_size,
            padding=self.conv_padding,
        )
        self.activation2 = torch.nn.ReLU()
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=self.pool_size)

        self.out_width = self.width // self.pool_size ** 2
        self.out_height = self.height // self.pool_size ** 2

        self.fc = torch.nn.Linear(
            in_features=self.out_width * self.conv_filters,
            out_features=self.out_height,
        )

        self.activation3 = torch.nn.ReLU()

        self.gru1 = torch.nn.GRU(
            input_size=self.out_height,
            hidden_size=self.rnn_size,
            bidirectional=True
        )

        self.gru2 = torch.nn.GRU(
            input_size=self.rnn_size,
            hidden_size=self.rnn_size,
            bidirectional=True
        )

        self.out_fc = torch.nn.Linear(
            in_features=self.rnn_size * 2,
            out_features=self.num_classes
        )

        self.out_activation = torch.nn.LogSoftmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.activation2(x)

        x = x.view(-1, self.out_height, self.out_width * self.conv_filters)

        x = self.fc(x)

        x, _ = self.gru1(x)
        split_x = torch.split(x, self.rnn_size, dim=2)
        x = torch.add(split_x[0], split_x[1])

        x, _ = self.gru2(x)
        split_x = torch.split(x, self.rnn_size, dim=2)
        x = torch.cat(split_x, dim=2)

        x = self.out_fc(x)
        x = self.out_activation(x)

        return x
