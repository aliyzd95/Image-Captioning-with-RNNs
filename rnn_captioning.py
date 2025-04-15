import math
from typing import Optional, Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction


def hello_rnn_captioning():
    print("Hello from rnn_captioning.py!")


class ImageEncoder(nn.Module):
    """
    Convolutional network that accepts images as input and outputs their spatial
    grid features. This module servesx as the image encoder in image captioning
    model. We will use a tiny RegNet-X 400MF model that is initialized with
    ImageNet-pretrained weights from Torchvision library.

    NOTE: We could use any convolutional network architecture, but we opt for a
    tiny RegNet model so it can train decently with a single K80 Colab GPU.
    """

    def __init__(self, pretrained: bool = True, verbose: bool = True):
        """
        Args:
            pretrained: Whether to initialize this model with pretrained weights
                from Torchvision library.
            verbose: Whether to log expected output shapes during instantiation.
        """
        super().__init__()
        self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)

        # Torchvision models return global average pooled features by default.
        # Our attention-based models may require spatial grid features. So we
        # wrap the ConvNet with torchvision's feature extractor. We will get
        # the spatial features right before the final classification layer.
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={"trunk_output.block4": "c5"}
        )
        # We call these features "c5", a name that may sound familiar from the
        # object detection assignment. :-)

        # Pass a dummy batch of input images to infer output shape.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))["c5"]
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print("For input images in NCHW format, shape (2, 3, 224, 224)")
            print(f"Shape of output c5 features: {dummy_out.shape}")

        # Input image batches are expected to be float tensors in range [0, 1].
        # However, the backbone here expects these tensors to be normalized by
        # ImageNet color mean/std (as it was trained that way).
        # We define a function to transform the input images before extraction:
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        """
        Number of output channels in extracted image features. You may access
        this value freely to define more modules to go with this encoder.
        """
        return self._out_channels

    def forward(self, images: torch.Tensor):
        # Input images may be uint8 tensors in [0-255], change them to float
        # tensors in [0-1]. Get float type from backbone (could be float32/64).
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0

        # Normalize images by ImageNet color mean/std.
        images = self.normalize(images)

        # Extract c5 features from encoder (backbone) and return.
        # shape: (B, out_channels, H / 32, W / 32)
        features = self.backbone(images)["c5"]
        return features


##############################################################################
# Recurrent Neural Network                                                   #
##############################################################################
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Args:
        x: Input data for this timestep, of shape (N, D).
        prev_h: Hidden state from previous timestep, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        next_h: Next hidden state, of shape (N, H)
        cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##########################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store next
    # hidden state and any values you need for the backward pass in the next_h
    # and cache variables respectively.
    ##########################################################################
    # Replace "pass" statement with your code
    
    preact = x @ Wx + prev_h @ Wh + b  # محاسبه پیش‌فعال‌سازی با جمع ضرب‌های ورودی و حالت قبلی
    next_h = torch.tanh(preact)        # اعمال تابع tanh برای به‌دست آوردن حالت مخفی بعدی
    cache = (x, Wx, prev_h, Wh, b, preact)  # ذخیره مقادیر لازم برای گام پس‌انتشار خطا

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Args:
        dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
        cache: Cache object from the forward pass

    Returns a tuple of:
        dx: Gradients of input data, of shape (N, D)
        dprev_h: Gradients of previous hidden state, of shape (N, H)
        dWx: Gradients of input-to-hidden weights, of shape (D, H)
        dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
        db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.
    #
    # HINT: For the tanh function, you can compute the local derivative in
    # terms of the output value from tanh.
    ##########################################################################
    # Replace "pass" statement with your code
    
    x, Wx, prev_h, Wh, b, preact = cache  # استخراج متغیرهای ذخیره‌شده

    # محاسبه مشتق نسبت به پیش‌فعال‌سازی (با مشتق tanh)
    d_preact = dnext_h * (1 - torch.tanh(preact)**2)

    # محاسبه گرادیان‌های بایاس، وزن‌ها، ورودی‌ها و وضعیت مخفی قبلی
    db = d_preact.sum(axis=0)
    dWh, dprev_h = prev_h.T @ d_preact, d_preact @ Wh.T
    dWx, dx = x.T @ d_preact, d_preact @ Wx.T

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Args:
        x: Input data for the entire timeseries, of shape (N, T, D).
        h0: Initial hidden state, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        h: Hidden states for the entire timeseries, of shape (N, T, H).
        cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##########################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of
    # input data. You should use the rnn_step_forward function that you defined
    # above. You can use a for loop to help compute the forward pass.
    ##########################################################################
    # Replace "pass" statement with your code
    
    # دریافت ابعاد داده‌ها
    N, T, D = x.shape  # N: تعداد دنباله‌ها، T: طول دنباله، D: ابعاد ورودی
    H = h0.shape[1]  # H: ابعاد وضعیت مخفی

    # مقداردهی اولیه وضعیت‌های مخفی و کش
    h = torch.zeros((N, T, H), dtype=x.dtype, device=x.device)  # وضعیت‌های مخفی
    cache = []  # کش برای ذخیره اطلاعات

    prev_h = h0  # وضعیت مخفی اولیه

    # حلقه برای پیش‌خور
    for t in range(T):
        xt = x[:, t, :]  # ورودی برای گام t
        next_h, step_cache = rnn_step_forward(xt, prev_h, Wx, Wh, b)  # محاسبه وضعیت مخفی
        h[:, t, :] = next_h  # ذخیره وضعیت مخفی
        prev_h = next_h  # به‌روزرسانی وضعیت مخفی قبلی
        cache.append(step_cache)  # ذخیره کش

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Args:
        dh: Upstream gradients of all hidden states, of shape (N, T, H).

    NOTE: 'dh' contains the upstream gradients produced by the
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
        dx: Gradient of inputs, of shape (N, T, D)
        dh0: Gradient of initial hidden state, of shape (N, H)
        dWx: Gradient of input-to-hidden weights, of shape (D, H)
        dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
        db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##########################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire
    # sequence of data. You should use the rnn_step_backward function that you
    # defined above. You can use a for loop to help compute the backward pass.
    ##########################################################################
    # Replace "pass" statement with your code
    
    # دریافت ابعاد گرادیان‌ها
    N, T, H = dh.shape  # ابعاد گرادیان
    D = cache[0][0].shape[1]  # ابعاد ورودی

    # مقداردهی اولیه گرادیان‌ها
    dx = torch.zeros((N, T, D), dtype=dh.dtype, device=dh.device)  # گرادیان ورودی
    dWx = torch.zeros((D, H), dtype=dh.dtype, device=dh.device)  # گرادیان وزن ورودی به حالت
    dWh = torch.zeros((H, H), dtype=dh.dtype, device=dh.device)  # گرادیان وزن حالت به حالت
    db = torch.zeros((H,), dtype=dh.dtype, device=dh.device)  # گرادیان بایاس

    dprev_tsh = dh[:, T-1, :]  # گرادیان وضعیت مخفی آخرین گام

    # حلقه برای پس‌انتشار
    for ts in range(T-1, -1, -1):
        out_backward = rnn_step_backward(dprev_tsh, cache[ts])  # اجرای پس‌انتشار
        dx[:, ts, :] = out_backward[0]  # ذخیره گرادیان ورودی
        dWx += out_backward[2]  # جمع کردن گرادیان وزن‌ها
        dWh += out_backward[3]
        db += out_backward[4]
        
        dprev_h = out_backward[1]  # گرادیان وضعیت مخفی
        dprev_tsh = dprev_h + dh[:, ts-1, :] if ts > 0 else dprev_h  # به‌روزرسانی گرادیان وضعیت مخفی قبلی
    dh0 = dprev_h  # گرادیان وضعیت مخفی اولیه


    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################
    return dx, dh0, dWx, dWh, db


class RNN(nn.Module):
    """
    Single-layer vanilla RNN module.

    You don't have to implement anything here but it is highly recommended to
    read through the code as you will implement subsequent modules.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize an RNN. Model parameters to initialize:
            Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
            Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
            b: Biases, of shape (H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        """
        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            h0: Initial hidden state, of shape (N, H)

        Returns:
            hn: The hidden state output
        """
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        """
        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)

        Returns:
            next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


class WordEmbedding(nn.Module):
    """
    Simplified version of torch.nn.Embedding.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Args:
        x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.

    Returns a tuple of:
        out: Array of shape (N, T, D) giving word vectors for all input words.
    """

    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()

        # Register parameters
        self.W_embed = nn.Parameter(
            torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
        )

    def forward(self, x):

        out = None
        ######################################################################
        # TODO: Implement the forward pass for word embeddings.
        ######################################################################
        # Replace "pass" statement with your code
                
        out = self.W_embed[x]  # استخراج بردارهای مربوط به ایندکس‌های ورودی

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        return out


def temporal_softmax_loss(x, y, ignore_index=None):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, *summing* the loss over all timesteps and *averaging* across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional ignore_index argument
    tells us which elements in the caption should not contribute to the loss.

    Args:
        x: Input scores, of shape (N, T, V)
        y: Ground-truth indices, of shape (N, T) where each element is in the
            range 0 <= y[i, t] < V

    Returns a tuple of:
        loss: Scalar giving loss
    """
    loss = None

    ##########################################################################
    # TODO: Implement the temporal softmax loss function.
    #
    # REQUIREMENT: This part MUST be done in one single line of code!
    #
    # HINT: Look up the function torch.functional.cross_entropy, set
    # ignore_index to the variable ignore_index (i.e., index of NULL) and
    # set reduction to either 'sum' or 'mean' (avoid using 'none' for now).
    #
    # We use a cross-entropy loss at each timestep, *summing* the loss over
    # all timesteps and *averaging* across the minibatch.
    ##########################################################################
    # Replace "pass" statement with your code
    
    loss = nn.functional.cross_entropy(torch.transpose(x, 1, 2), y,
                          ignore_index=ignore_index, reduction='sum') / x.shape[0]
    
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return loss


class CaptioningRNN(nn.Module):
    """
    A CaptioningRNN produces captions from images using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.

    You will implement the `__init__` method for model initialization and
    the `forward` method first, then come back for the `sample` method later.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim: int = 512,
        wordvec_dim: int = 128,
        hidden_dim: int = 128,
        cell_type: str = "rnn",
        image_encoder_pretrained: bool = True,
        ignore_index: Optional[int] = None,
    ):
        """
        Construct a new CaptioningRNN instance.

        Args:
            word_to_idx: A dictionary giving the vocabulary. It contains V
                entries, and maps each string to a unique integer in the
                range [0, V).
            input_dim: Dimension D of input image feature vectors.
            wordvec_dim: Dimension W of word vectors.
            hidden_dim: Dimension H for the hidden state of the RNN.
            cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        """
        super().__init__()
        if cell_type not in {"rnn", "lstm", "attn"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        self.ignore_index = ignore_index

        ######################################################################
        # TODO: Initialize the image captioning module. Refer to the TODO
        # in the captioning_forward function on layers you need to create
        #
        # You may want to check the following pre-defined classes:
        # ImageEncoder WordEmbedding, RNN, LSTM, AttentionLSTM, nn.Linear
        #
        # (1) output projection (from RNN hidden state to vocab probability)
        # (2) feature projection (from CNN pooled feature to h0)
        ######################################################################
        # Replace "pass" statement with your code
        
        # مقداردهی اولیه لایه رمزگذار تصویر
        self.image_encoder_layer = ImageEncoder(pretrained=image_encoder_pretrained)  # (N, C, 4, 4)

        # اگر از LSTM با توجه استفاده می‌کنیم
        if self.cell_type == "attn":
            # لایه کانولوشن برای استخراج ویژگی‌ها
            self.conv = nn.Conv2d(self.image_encoder_layer.out_channels, hidden_dim, kernel_size=1, stride=1)  # (N, H, 4, 4)
        else:
            # اگر از RNN یا LSTM استفاده می‌کنیم
            self.avg_pool_layer = nn.AvgPool2d(kernel_size=4, stride=4)  # کاهش ابعاد ویژگی‌ها به (N, C)
            self.transform_layer = nn.Linear(self.image_encoder_layer.out_channels, hidden_dim)  # تبدیل ابعاد ویژگی‌ها به (N, H)

        # لایه تعبیه‌سازی کلمات
        self.word_embedding_layer = WordEmbedding(vocab_size, wordvec_dim)

        # انتخاب نوع RNN
        if self.cell_type == "rnn":
            self.rnn_layer = RNN(wordvec_dim, hidden_dim)  # لایه RNN
        elif self.cell_type == "lstm":
            self.lstm_layer = LSTM(wordvec_dim, hidden_dim)  # لایه LSTM
        elif self.cell_type == "attn":
            self.attn_lstm_layer = AttentionLSTM(wordvec_dim, hidden_dim)  # لایه LSTM با توجه

        # لایه خطی برای پیش‌بینی توزیع واژه‌ها
        self.linear_layer = nn.Linear(hidden_dim, vocab_size)  # تبدیل وضعیت پنهان به توزیع احتمال واژه‌ها

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def forward(self, images, captions):
        """
        Compute training-time loss for the RNN. We input images and the GT
        captions for those images, and use an RNN (or LSTM) to compute loss. The
        backward part will be done by torch.autograd.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            captions: Ground-truth captions; an integer array of shape (N, T + 1)
                where each element is in the range 0 <= y[i, t] < V

        Returns:
            loss: A scalar loss
        """
        # Cut captions into two pieces: captions_in has everything but the last
        # word and will be input to the RNN; captions_out has everything but the
        # first word and this is what we will expect the RNN to generate. These
        # are offset by one relative to each other because the RNN should produce
        # word (t+1) after receiving word t. The first element of captions_in
        # will be the START token, and the first element of captions_out will
        # be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0
        ######################################################################
        # TODO: Implement the forward pass for the CaptioningRNN.
        # In the forward pass you will need to do the following:
        # (1) Use an affine transformation to project the image feature to
        #     the initial hidden state $h0$ (for RNN/LSTM, of shape (N, H)) or
        #     the projected CNN activation input $A$ (for Attention LSTM,
        #     of shape (N, H, 4, 4).
        # (2) Use a word embedding layer to transform the words in captions_in
        #     from indices to vectors, giving an array of shape (N, T, W).
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to
        #     process the sequence of input word vectors and produce hidden state
        #     vectors for all timesteps, producing an array of shape (N, T, H).
        # (4) Use a (temporal) affine transformation to compute scores over the
        #     vocabulary at every timestep using the hidden states, giving an
        #     array of shape (N, T, V).
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring
        #     the points where the output word is <NULL>.
        #
        # Do not worry about regularizing the weights or their gradients!
        ######################################################################
        # Replace "pass" statement with your code
        
        # استخراج ویژگی‌های تصویر از لایه رمزگذار تصویر
        images = self.image_encoder_layer(images)  # (N, H)

        # بسته به نوع RNN، ویژگی‌های تصویر را پردازش می‌کنیم
        if self.cell_type == "attn":
            images = self.conv(images)  # برای LSTM با توجه
        else:
            images = self.avg_pool_layer(images).squeeze()  # کاهش ابعاد ویژگی‌ها
            images = self.transform_layer(images)  # تبدیل ابعاد ویژگی‌ها به وضعیت پنهان

        # تبدیل کلمات ورودی به بردارهای تعبیه‌سازی‌شده
        captions_in = self.word_embedding_layer(captions_in)

        # h0 وضعیت پنهان اولیه است که از ویژگی‌های تصویر به دست آمده
        if self.cell_type == "rnn":
            hidden_states = self.rnn_layer(captions_in, images)  # پردازش با RNN
        elif self.cell_type == "lstm":
            hidden_states = self.lstm_layer(captions_in, images)  # پردازش با LSTM
        elif self.cell_type == "attn":
            hidden_states = self.attn_lstm_layer(captions_in, images)  # پردازش با LSTM با توجه

        # محاسبه نمرات برای هر کلمه در دایره واژگان
        hidden_states = self.linear_layer(hidden_states)

        # محاسبه خطای نرم‌افزاری با استفاده از نمرات و کلمات واقعی
        loss = temporal_softmax_loss(hidden_states, captions_out, ignore_index=self.ignore_index)

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return loss

    def sample(self, images, max_length=15):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            max_length: Maximum length T of generated captions

        Returns:
            captions: Array of shape (N, max_length) giving sampled captions,
                where each element is an integer in the range [0, V). The first
                element of captions should be the first sampled word, not the
                <START> token.
        """
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == "attn":
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()

        ######################################################################
        # TODO: Implement test-time sampling for the model. You will need to
        # initialize the hidden state of the RNN by applying the learned affine
        # transform to the image features. The first word that you feed to
        # the RNN should be the <START> token; its value is stored in the
        # variable self._start. At each timestep you will need to do to:
        # (1) Embed the previous word using the learned word embeddings
        # (2) Make an RNN step using the previous hidden state and the embedded
        #     current word to get the next hidden state.
        # (3) Apply the learned affine transformation to the next hidden state to
        #     get scores for all words in the vocabulary
        # (4) Select the word with the highest score as the next word, writing it
        #     (the word index) to the appropriate slot in the captions variable
        #
        # For simplicity, you do not need to stop generating after an <END> token
        # is sampled, but you can if you want to.
        #
        # NOTE: we are still working over minibatches in this function. Also if
        # you are using an LSTM, initialize the first cell state to zeros.
        # For AttentionLSTM, first project the 1280x4x4 CNN feature activation
        # to $A$ of shape Hx4x4. The LSTM initial hidden state and cell state
        # would both be A.mean(dim=(2, 3)).
        #######################################################################
        # Replace "pass" statement with your code
        
        # استخراج ویژگی‌های تصویر از لایه رمزگذار
        images = self.image_encoder_layer(images)

        # برای LSTM با توجه، وضعیت‌های پنهان و وضعیت سلولی اولیه را محاسبه می‌کنیم
        if self.cell_type == "attn":
            images = self.conv(images)  # پردازش ویژگی‌ها با لایه کانولوشن
            prev_h = images.mean(dim=(2, 3))  # میانگین گیری برای وضعیت پنهان
            prev_c = images.mean(dim=(2, 3))  # وضعیت سلولی اولیه

        else:
            images = self.avg_pool_layer(images).squeeze()  # کاهش ابعاد ویژگی‌ها
            prev_h = self.transform_layer(images)  # تبدیل به وضعیت پنهان اولیه

        # کلمه اولیه را با استفاده از نشانه <START> تعبیه‌سازی می‌کنیم
        x = self.word_embedding_layer([self._start for _ in range(N)])  

        # برای LSTM، وضعیت سلولی اولیه را صفر می‌کنیم
        if self.cell_type == "lstm":
            prev_c = torch.zeros_like(prev_h)

        # تولید زیرنویس با استفاده از RNN
        for t in range(max_length):
            if self.cell_type == "rnn":
                prev_h = self.rnn_layer.step_forward(x, prev_h)  # مرحله بعدی RNN

            elif self.cell_type == "lstm":
                prev_h, prev_c = self.lstm_layer.step_forward(x, prev_h, prev_c)  # مرحله بعدی LSTM

            elif self.cell_type == "attn":
                attn, attn_weights = dot_product_attention(prev_h, images)  # محاسبه توجه
                attn_weights_all[:, t] = attn_weights  # ذخیره وزن‌های توجه
                prev_h, prev_c = self.attn_lstm_layer.step_forward(x, prev_h, prev_c, attn)  # مرحله بعدی LSTM با توجه

            # محاسبه نمرات برای هر کلمه در دایره واژگان
            scores = self.linear_layer(prev_h)
            _, idx = scores.max(dim=1)  # انتخاب کلمه با بالاترین نمره
            x = self.word_embedding_layer(idx)  # تعبیه‌سازی کلمه انتخاب‌شده
            captions[:, t] = idx  # ذخیره کلمه در زیرنویس

        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################
        if self.cell_type == "attn":
            return captions, attn_weights_all.cpu()
        else:
            return captions

