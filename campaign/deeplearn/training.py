# -*- coding: utf-8 -*-
import click
import numpy as np
import pandas as pd
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import Latest
from zipline.pipeline.factors import Returns
from zipline.utils.cli import Date

from me.helper.research_env import Research
from me.pipeline.factors.tsfactor import Fundamental

pd.set_option('display.width', 800)


def make_pipeline(asset_finder):
    h2o = USEquityPricing.high.latest / USEquityPricing.open.latest
    l2o = USEquityPricing.low.latest / USEquityPricing.open.latest
    c2o = USEquityPricing.close.latest / USEquityPricing.open.latest
    h2c = USEquityPricing.high.latest / USEquityPricing.close.latest
    l2c = USEquityPricing.low.latest / USEquityPricing.close.latest
    h2l = USEquityPricing.high.latest / USEquityPricing.low.latest

    vol = USEquityPricing.volume.latest
    outstanding = Fundamental(asset_finder).outstanding
    outstanding.window_safe = True
    turnover_rate = vol / Latest([outstanding])
    returns = Returns(inputs=[USEquityPricing.close], window_length=5)  # 预测一周数据

    pipe_columns = {
        'h2o': h2o.log1p().zscore(),
        'l2o': l2o.log1p().zscore(),
        'c2o': c2o.log1p().zscore(),
        'h2c': h2c.log1p().zscore(),
        'l2c': l2c.log1p().zscore(),
        'h2l': h2l.log1p().zscore(),
        'vol': vol.zscore(),
        'turnover_rate': turnover_rate.log1p().zscore(),
        'return': returns.log1p(),
    }
    # pipe_screen = (low_returns | high_returns)
    pipe = Pipeline(columns=pipe_columns)
    return pipe


def make_input(result):
    # start = '2014-9-1'
    # end   = '2017-9-11'
    P = result.pivot_table(index=result['level_0'], columns='level_1',
                           values=['h2o', 'l2o', 'c2o', 'h2c', 'l2c', 'vol', 'turnover_rate',
                                   'return'])  # Make a pivot table from the data
    mi = P.columns.tolist()

    new_ind = pd.Index(e[1].symbol + '_' + e[0] for e in mi)
    P.columns = new_ind
    P = P.sort_index(axis=1)  # Sort by columns
    P.index.name = 'date'
    # clean_and_flat = P.dropna(1) #去掉0列
    clean_and_flat = P  # 去掉0列

    print clean_and_flat
    print "*" * 50, "flat result", "*" * 50

    target_cols = list(filter(lambda x: 'return' in x, clean_and_flat.columns.values))
    input_cols = list(filter(lambda x: 'return' not in x, clean_and_flat.columns.values))
    size = len(clean_and_flat)
    InputDF = clean_and_flat[input_cols][:size]
    TargetDF = clean_and_flat[target_cols][:size]
    return InputDF, TargetDF


#############################################################################################################

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat

learn = tf.contrib.learn

RNN_HIDDEN_SIZE = 100  # 我们寄希望发现100个隐藏特征
NUM_LAYERS = 2
BATCH_SIZE = 21  # PARM
NUM_EPOCHS = 200  # 200   #PARM
lr = 0.003


def train(InputDF, TargetDF):
    print "*" * 50, "Training a rnn network", "*" * 50
    num_features = len(InputDF.columns)
    num_stocks = len(TargetDF.columns)
    print "num stocks %s,last train data %s,first train data %s" % (num_stocks, TargetDF.index[-1], TargetDF.index[0])

    # 生成数据
    used_size = (len(InputDF) // BATCH_SIZE) * BATCH_SIZE  # 要BATCH_SIZE整数倍
    train_X, train_y = InputDF[-used_size:].values, TargetDF[-used_size:].values
    test_X, test_y = InputDF[-BATCH_SIZE:].values, TargetDF[-BATCH_SIZE:].values  # TODO
    train_X = train_X.astype(np.float32)
    train_y = train_y.astype(np.float32)
    test_X = test_X.astype(np.float32)
    test_y = test_y.astype(np.float32)
    print np.shape(train_X), np.shape(train_y)
    print "Train Set <X:y> shape"
    print "Train Data Count:%s , Feather Count:%s , Stock Count:%s" % (
        len(train_X), num_features, num_stocks)

    NUM_TRAIN_BATCHES = int(len(train_X) / BATCH_SIZE)
    ATTN_LENGTH = 10
    dropout_keep_prob = 0.5

    def LstmCell():
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_HIDDEN_SIZE, state_is_tuple=True)
        return lstm_cell

    def makeGRUCells():
        cells = []
        for i in range(NUM_LAYERS):
            cell = tf.nn.rnn_cell.GRUCell(num_units=RNN_HIDDEN_SIZE)
            if len(cells) == 0:
                # Add attention wrapper to first layer.
                cell = tf.contrib.rnn.AttentionCellWrapper(
                    cell, attn_length=ATTN_LENGTH, state_is_tuple=True)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
            cells.append(cell)
        attn_cell = tf.nn.rnn_cell.MultiRNNCell(cells,
                                                state_is_tuple=True)  # GRUCell必须false，True 比错 ,如果是BasicLSTMCell 必须True
        return attn_cell

    def lstm_model(X, y):
        cell = makeGRUCells()
        ''' #理论dynnamic rnn 首选，但计算速度相比静态慢很多，不知何故
       output, _ = tf.nn.dynamic_rnn(
                                      cell,
                                      inputs=tf.expand_dims(X, -1),
                                      dtype=tf.float32,
                                      time_major=False
                                      )
        '''
        split_inputs = tf.reshape(X, shape=[1, BATCH_SIZE,
                                            num_features])  # Each item in the batch is a time step, iterate through them
        # print split_inputs
        split_inputs = tf.unstack(split_inputs, axis=1, name="unpack_l1")
        output, _ = tf.nn.static_rnn(cell,
                                     inputs=split_inputs,
                                     dtype=tf.float32
                                     )

        output = tf.transpose(output, [1, 0, 2])
        output = output[-1]
        # 通过无激活函数的全连接层,计算就是线性回归，并将数据压缩成一维数组结构
        predictions = tf.contrib.layers.fully_connected(output, num_stocks, None)
        labels = y
        loss = tf.losses.mean_squared_error(predictions, labels)
        # print "lost:",loss
        train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
                                                   optimizer="Adagrad",
                                                   learning_rate=lr)
        return predictions, loss, train_op

    PRINT_STEPS = 100
    validation_monitor = learn.monitors.ValidationMonitor(test_X, test_y,
                                                          every_n_steps=PRINT_STEPS,
                                                          early_stopping_rounds=1000)
    # 进行训练
    regressor = SKCompat(learn.Estimator(model_fn=lstm_model,
                                         # model_dir="Models/model_0",
                                         config=tf.contrib.learn.RunConfig(
                                             save_checkpoints_steps=100,
                                             save_checkpoints_secs=None,
                                             save_summary_steps=100,
                                         )))

    print "Total Train Step: ", NUM_TRAIN_BATCHES * NUM_EPOCHS
    print "*" * 50, "Training a rnn regress task now", "*" * 50
    regressor.fit(train_X, train_y, batch_size=BATCH_SIZE,
                  steps=NUM_TRAIN_BATCHES * NUM_EPOCHS)  # steps=train_labels.shape[0]/batch_size * epochs,

    print "*" * 50, "Predict tomorrow stock price now", "*" * 50
    pred = regressor.predict(test_X[-BATCH_SIZE:])  # 使用最后21天预测　未来５天的股票价格

    return pred


def save_to_sqlite(date, pred, target_cols):
    print date, target_cols
    print pred
    date = date  # +1?  +5? 获取最后一天的数据
    df = pd.DataFrame(pred[-1:], index=[date], columns=target_cols)  # 所有的股票数据一次性预测
    df.index.name = "date"
    print "*" * 50, "Predicted stock price from last trade day", "*" * 50
    print df
    df.to_csv("predict.csv", encoding="utf-8")


@click.command()
@click.option(
    '-s',
    '--start',
    type=Date(tz='utc', as_timestamp=True),
    help='The start date of the train.',
)
@click.option(
    '-e',
    '--end',
    type=Date(tz='utc', as_timestamp=True),
    help='The start date of the train.',
)
def execute(start, end):
    research = Research()
    my_pipe = make_pipeline(research.get_engine()._finder)
    result = research.run_pipeline(my_pipe, start, end)

    InputDF, TargetDF = make_input(result.reset_index())
    predict = train(InputDF.fillna(0), TargetDF.fillna(0))
    save_to_sqlite(end, predict, TargetDF.columns)
    # save_to_sqlite(end,TargetDF[-1:].values,TargetDF.columns)


if __name__ == '__main__':
    print "Let's go ................. "
    execute()
