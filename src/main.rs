use mnist::*;
use ndarray::{s, Array2};
use rand::random;



fn main() {
    let (train_data, train_labels, test_data, test_labels) = load_data();


    let mut nn = NeuralNetwork::new(28 * 28, 10, 2, 16);

    let output = nn.network(&Array2::from_shape_vec((28 * 28, 1), vec![0.0; 28 * 28]).unwrap());
    //println!("Result: {}", output.t());

    let output = nn.network(&Array2::from_shape_vec((28 * 28, 1), vec![1.0; 28 * 28]).unwrap());
    //println!("Result: {}", output.t());

    let output = nn.network(&Array2::from_shape_vec((28 * 28, 1), vec![10.0; 28 * 28]).unwrap());
    //println!("Result: {}", output.t());

    // tests
    for i in 0..0 {
        let data = test_data.slice(s![.., i]).to_owned().insert_axis(ndarray::Axis(1));
        println!("data: {}", data.t());

        println!("Test label: {}", test_labels[[0, i]] as i32);
        let output = nn.network(&data);
        println!("Result: {}", output.t());
    }

    // repeated training on one image
    let i = 1;
    let data = train_data.slice(s![.., i]).to_owned().insert_axis(ndarray::Axis(1));
    let label = train_labels[[0, i]];
    println!("Label: {}", label as usize);
    for _ in 0..0000 {
        //println!("data: {}", data.t());

        let _output = nn.network_train(&data, label, 0.5);
    }

    // training only on some numbers
    for i in 0..5000 {
        let label = train_labels[[0, i]];
        if label >= 3.0 {
            continue;
        }

        let data = train_data.slice(s![.., i]).to_owned().insert_axis(ndarray::Axis(1));
        //println!("data: {}", data.t());

        let _output = nn.network_train(&data, label, 0.01);
    }

    // training
    for i in 0..0000 {
        let data = train_data.slice(s![.., i]).to_owned().insert_axis(ndarray::Axis(1));
        //println!("data: {}", data.t());

        let _output = nn.network_train(&data, train_labels[[0, i]], 0.01);
    }

    // print network
    //for layer in &nn.layers {
    //    println!("bias: {}", layer.bias.t());
    //    println!("weights: {}", layer.weights);
    //}
    
    // tests
    for i in 0..1 {
        let data = test_data.slice(s![.., i]).to_owned().insert_axis(ndarray::Axis(1));
        //println!("data: {}", data.t());

        println!("Test label: {}", test_labels[[0, i]] as i32);
        let output = nn.network(&data);
        println!("Result: {}", output.t());
    }

    for i in 5000..00000 {
        let data = train_data.slice(s![.., i]).to_owned().insert_axis(ndarray::Axis(1));
        //println!("data: {}", data.t());

        let _output = nn.network_train(&data, train_labels[[0, i]], 0.01);
    }

    // tests
    for i in 1..20 {
        let data = test_data.slice(s![.., i]).to_owned().insert_axis(ndarray::Axis(1));
        //println!("data: {}", data.t());

        println!("Test label: {}", test_labels[[0, i]] as i32);
        let output = nn.network(&data);
        println!("Result: {}", output.t());
    }
}




struct NeuralNetwork {
    layers: Vec<Layer>
}

struct Layer {
    weights: Array2<f32>,   // must have as many columns as neurons on the previous layer and as many rows as this layer
    bias: Array2<f32>,      // must have as many rows as in weights and only one column
}

impl NeuralNetwork {
    fn new(inputs: usize, outputs: usize, hidden_layers: usize, hidden_layers_size: usize) -> Self {
        let mut layers = Vec::new();

        // first layer (hidden)
        let random_weights: Vec<f32> = (0..(hidden_layers_size * inputs)).map(|_| rand()).collect();
        let random_bias: Vec<f32> = (0..hidden_layers_size).map(|_| rand()).collect();
        //let random_bias: Vec<f32> = vec![0.0; hidden_layers_size];
        layers.push(Layer {
            weights: Array2::from_shape_vec((hidden_layers_size, inputs), random_weights).unwrap(),
            bias: Array2::from_shape_vec((hidden_layers_size, 1), random_bias).unwrap()
        });

        // hidden layers
        for _ in 1..hidden_layers {
            let random_weights: Vec<f32> = (0..(hidden_layers_size * hidden_layers_size)).map(|_| rand()).collect();
            let random_bias: Vec<f32> = (0..hidden_layers_size).map(|_| rand()).collect();
            //let random_bias: Vec<f32> = vec![0.0; hidden_layers_size];
            layers.push(Layer {
                weights: Array2::from_shape_vec((hidden_layers_size, hidden_layers_size), random_weights).unwrap(),
                bias: Array2::from_shape_vec((hidden_layers_size, 1), random_bias).unwrap()
            });
        }

        // output layer
        let random_weights: Vec<f32> = (0..(outputs * hidden_layers_size)).map(|_| rand()).collect();
        let random_bias: Vec<f32> = (0..outputs).map(|_| rand()).collect();
        //let random_bias: Vec<f32> = vec![0.0; outputs];
        layers.push(Layer {
            weights: Array2::from_shape_vec((outputs, hidden_layers_size), random_weights).unwrap(),
            bias: Array2::from_shape_vec((outputs, 1), random_bias).unwrap(),
        });

        Self { layers }
    }

    fn network(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = neuron_layer(&output, &layer.weights, &layer.bias);
            output = sigmoid_layer(&output);
        }
        output
    }

    fn network_train(&mut self, input: &Array2<f32>, label: f32, learning_rate: f32) -> Array2<f32> {
        let mut output = input.clone();

        let mut outputs = Vec::new();
        let mut activations = Vec::new();

        // put the input in activations[0]
        activations.push(input.clone());

        // forward propagation
        for layer in &self.layers {
            output = neuron_layer(&output, &layer.weights, &layer.bias);
            outputs.push(output.clone());

            output = sigmoid_layer(&output);
            activations.push(output.clone());
        }
        
        // backward propagation
        let _cost = cost(&output, label);
        let mut cost_derivative = cost_derivative(&output, label);
        for i in (0..self.layers.len()).rev() {
            //println!("Shape activations: {:?}", activations[i].shape());
            //println!("Shape cost_derivative: {:?}", cost_derivative.shape());
            //println!("Shape weights: {:?}", self.layers[i].weights.shape());
            //println!("Shape weights: {:?}", self.layers[i].weights.shape());
            //println!("Shape weights: {:?}", self.layers[i].weights.shape());
            //println!("Shape weights: {:?}", self.layers[i].weights.shape());
            
            // calculate how weights should be changed
            let delta = sigmoid_derivative_layer(&outputs[i]) * &cost_derivative;
            //println!("Shape delta: {:?}", delta.shape());
            let dcost_dweight = delta.dot(&activations[i].t());
            //let dcost_dweight = activations[i].dot(&delta.t());
            //println!("Shape dcost_dweight: {:?}", dcost_dweight.shape());
            
            // change bias
            self.layers[i].bias = &self.layers[i].bias - delta.sum_axis(ndarray::Axis(0)) * learning_rate;
            //println!("Shape bias: {:?}", self.layers[i].bias.shape());
            
            // change weights
            self.layers[i].weights = &self.layers[i].weights - &dcost_dweight * learning_rate;
            
            // save cost_derivative
            cost_derivative = self.layers[i].weights.t().dot(&delta);
            //println!("bruh {i}");
            //println!("broo {i}");
            //println!("Shape cost_derivative: {:?}", cost_derivative.shape());
        }

        output
    }
}

fn neuron_layer(input: &Array2<f32>, weights: &Array2<f32>, bias: &Array2<f32>) -> Array2<f32> {
    weights.dot(input) + bias
}

fn sigmoid_layer(input: &Array2<f32>) -> Array2<f32> {
    input.mapv(|x| sigmoid(x))
}

fn sigmoid_derivative_layer(input: &Array2<f32>) -> Array2<f32> {
    input.mapv(|x| sigmoid_derivative(x))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f32) -> f32 {
    let sig = sigmoid(x);
    sig * (1.0 - sig)
}

fn cost(input: &Array2<f32>, label: f32) -> Array2<f32> {
    let mut label_vec = Array2::zeros((10, 1));
    label_vec[[label as usize, 0]] = 1.0;

    (input - label_vec).mapv(|x| x * x)
}

fn cost_derivative(input: &Array2<f32>, label: f32) -> Array2<f32> {
    let mut label_vec = Array2::zeros((10, 1));
    label_vec[[label as usize, 0]] = 1.0;

    2.0 * (input - label_vec)
}

fn rand() -> f32 {
    random::<f32>() * 2.0 - 1.0
}

fn load_data() -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    // https://docs.rs/mnist/latest/mnist/
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array2::from_shape_vec((28 * 28, 50_000), trn_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f32 / 256.0);
    //println!("{:#.1?}\n",train_data.slice(s![image_num, ..]));

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f32> = Array2::from_shape_vec((1, 50_000), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f32);
    println!("The first digit is a {:?}",train_labels.slice(s![image_num, ..]) );

    let _test_data = Array2::from_shape_vec((28 * 28, 10_000), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f32 / 256.0);

    let _test_labels: Array2<f32> = Array2::from_shape_vec((1, 10_000), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f32);

    (train_data, train_labels, _test_data, _test_labels)
}