import itertools
import pickle
from collections import OrderedDict

import tensorflow as tf

from nlp.entity.loader import *
from nlp.entity.model import Model

flags = tf.app.flags
flags.DEFINE_boolean("clean", False, "clean train folder")
flags.DEFINE_boolean("train", False, "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
flags.DEFINE_integer("lstm_dim", 100, "Num of hidden units in LSTM, or num of filters in IDCNN")
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip", 5, "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("batch_size", 20, "batch size")
flags.DEFINE_float("lr", 0.001, "Initial learning rate")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
flags.DEFINE_boolean("pre_emb", True, "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros", False, "Wither replace digits with zero")
flags.DEFINE_boolean("lower", True, "Wither lower case")

flags.DEFINE_integer("max_epoch", 100, "maximum training epochs")
flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")
flags.DEFINE_string("ckpt_path", "ckpt", "Path to save model")
flags.DEFINE_string("summary_path", "summary", "Path to store summaries")
flags.DEFINE_string("log_file", "train.log", "File for log")
flags.DEFINE_string("map_file", 'data/entity/maps.pkl', "file for maps")
flags.DEFINE_string("vocab_file", "vocab.json", "File for vocab")
flags.DEFINE_string("config_file", 'data/entity/config_file', "File for config")
flags.DEFINE_string("script", "conlleval", "evaluation script")
flags.DEFINE_string("result_path", "result", "Path for results")
flags.DEFINE_string("emb_file", 'data/entity/train/vec.txt', "Path for pre_trained embedding")
flags.DEFINE_string("train_file", 'data/entity/train/example.train', "Path for train data")
flags.DEFINE_string("dev_file", 'data/entity/train/example.dev', "Path for dev data")
flags.DEFINE_string("test_file", 'data/entity/train/example.test', "Path for test data")

flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
# flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag):
    print("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        print(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            print("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            print("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    # load data sets
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)

    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        if FLAGS.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    train_manager = BatchManager(train_data, FLAGS.batch_size)
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char)
        print("start training")
        loss = []
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    print("iteration:{} step:{}/{}, ""NER loss:{:>9.6f}".format(
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag)
            if best:
                save_model(sess, model, FLAGS.ckpt_path)
            evaluate(sess, model, "test", test_manager, id_to_tag)


def evaluate_line():
    config = load_config(FLAGS.config_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, False)
        while True:
            # try:
            #     line = input("请输入测试句子:")
            #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            #     print(result)
            # except Exception as e:
            #     logger.info(e)

            line = input("请输入测试句子:")
            result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
            print(result)


def create_model(session, path, load_vec, config, id_to_char, is_train=True):
    # create model, reuse parameters if exists
    tf_model = Model(config, is_train)

    ckpt = tf.train.get_checkpoint_state(path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        tf_model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if config["pre_emb"]:
            emb_weights = session.run(tf_model.char_lookup.read_value())
            emb_weights = load_vec(config["emb_file"], id_to_char, config["char_dim"], emb_weights)
            session.run(tf_model.char_lookup.assign(emb_weights))
            print("Load pre-trained embedding.")
    return tf_model


def extract_entity(text_content):
    """
    提取文本中的实体
    :param text_content: 文本内容
    :return: 实体列表
    """
    config = load_config(FLAGS.config_file)
    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with open(FLAGS.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # 重置一下，否则下次运行会报有参数找不到
    tf.reset_default_graph()

    ckpt_path = 'data/entity/ckpt_biLSTM'
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, ckpt_path, load_word2vec, config, id_to_char, False)
        result = model.evaluate_line(sess, input_from_line(text_content, char_to_id), id_to_tag)
        origin_entities = result['entities']

    # 输出结果
    entities = set()
    for item in origin_entities:
        word = item['word'].strip()
        if len(word) > 1:
            entities.add(word)
    return list(entities)


class EntityExtraction:
    def __init__(self):
        config = load_config(FLAGS.config_file)
        # limit GPU memory
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with open(FLAGS.map_file, "rb") as file:
            self.char_to_id, id_to_char, tag_to_id, self.id_to_tag = pickle.load(file)

        ckpt_path = 'data/entity/ckpt_biLSTM'
        self.session = tf.Session(config=tf_config)
        self.entity_model = create_model(self.session, ckpt_path, load_word2vec, config, id_to_char, False)

    def extract_entity(self, text):
        """
        提取文本中的实体
        :param text: 文本
        :return: 实体列表
        """
        result = self.entity_model.evaluate_line(self.session, input_from_line(text, self.char_to_id), self.id_to_tag)
        if result is None:
            return []

        entities = set()
        for item in result['entities']:
            word = item['word'].strip()
            if len(word) > 1:
                entities.add(word)
        return list(entities)
