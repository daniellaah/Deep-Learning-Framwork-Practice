from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# 标题输入：接收一个含有 100 个整数的序列，每个整数在 1 到 10000 之间。
# 注意我们可以通过传递一个 `name` 参数来命名任何层。
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# Embedding 层将输入序列编码为一个稠密向量的序列，每个向量维度为 512。
x = Embedding(input_dim=10000, output_dim=512)(main_input)

# LSTM 层把向量序列转换成单个向量，它包含整个序列的上下文信息
lstm_out = LSTM(32)(x)

# 在这里，我们插入辅助损失，使得即使在模型主损失很高的情况下，LSTM 层和 Embedding 层都能被平稳地训练
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

# 此时，我们将辅助输入数据与 LSTM 层的输出连接起来，输入到模型中
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, auxiliary_input],
              outputs=[main_input, auxiliary_output])

# 现在编译模型，并给辅助损失分配一个 0.2 的权重
model.compile(optimizer='rmsprop',
              loss='binaray_crossentropy',
              loss_weights=[1, 0.2])
model.fit([main_data, additional_data], [lables, labels], epochs=50, batch_size=32)

# 由于输入和输出均被命名了（在定义时传递了一个 name 参数），我们也可以通过以下方式编译模型
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binaray_crossentropy',
                    'aux_output': 'binaray_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})
model.fit({'main_input': main_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output':labels},
          epochs=50, batch_size=32)
