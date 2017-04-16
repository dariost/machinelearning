pub struct Adaline
{
    weight: Vec<f64>,
    learning_rate: f64,
    iterations: usize,
}

impl Adaline
{
    pub fn new(num_features: usize, learning_rate: f64, iterations: usize) -> Adaline
    {
        Adaline {
            weight: vec![0.0; num_features + 1],
            learning_rate: learning_rate,
            iterations: iterations,
        }
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    pub fn fit(&mut self, data: &[&[f64]], class: &[bool], callback: Option<fn(&usize, &f64, &[f64])>) -> f64
    {
        let class: Vec<f64> = class.iter().map(|&x| if x {1.0} else {-1.0}).collect();
        assert_eq!(data.len(), class.len());
        assert!(data.iter().all(|x| x.len() == self.weight.len() - 1));
        let mut error = 0.0;
        for it in 0..self.iterations
        {
            let output: Vec<f64> = data.iter().map(|x| self.activation(x)).collect();
            let errors: Vec<f64> = class.iter().zip(output).map(|x| x.0 - x.1).collect();
            self.weight[0] += self.learning_rate * errors.iter().sum::<f64>();
            for feature in 1..self.weight.len()
            {
                let mut feature_error = 0.0;
                for i in 0..data.len()
                {
                    feature_error += data[i][feature - 1] * errors[i];
                }
                self.weight[feature] += self.learning_rate * feature_error;
            }
            error = errors.iter().map(|x| x * x).sum();
            if let Some(cb) = callback
            {
                cb(&it, &error, &self.weight);
            }
            if error == 0.0
            {
                break;
            }
        }
        error
    }

    fn activation(&self, data: &[f64]) -> f64
    {
        assert_eq!(data.len(), self.weight.len() - 1);
        self.weight
            .iter()
            .skip(1)
            .zip(data.iter())
            .map(|x| x.0 * x.1)
            .sum::<f64>() + self.weight[0]
    }

    pub fn predict(&self, data: &[f64]) -> bool
    {
        assert_eq!(data.len(), self.weight.len() - 1);
        self.activation(data) >= 0.0
    }
}

#[allow(dead_code)]
fn print_callback(iteration: &usize, errors: &f64, weight: &[f64])
{
    println!("Iteration #{}: {} error", iteration, errors);
    println!("Weight: {:?}", weight);
}

#[test]
fn trivial()
{
    let mut p = Adaline::new(2, 0.001, 10000);
    assert!(p.fit(&[&[1.0, 0.0], &[4.0, 1.0], &[0.0, 1.0], &[2.0, 3.0]], &[true, false, true, false], Some(print_callback)) <=
            0.01);
    assert_eq!(p.predict(&[-1.0, -1.0]), true);
    assert_eq!(p.predict(&[10.0, 10.0]), false);
    assert_eq!(p.predict(&[0.5, 0.5]), true);
}

#[test]
#[should_panic]
fn trivial_wrong()
{
    let mut p = Adaline::new(2, 0.001, 10000);
    assert!(p.fit(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[0.0, 0.0]], &[true, true, false, false], None) <= 0.01);
}

#[test]
fn trivial_3d()
{
    let mut p = Adaline::new(3, 0.001, 10000);
    assert!(p.fit(&[&[1.0, 0.0, 1.0],
                    &[0.0, 1.0, 1.0],
                    &[1.0, 1.0, 0.0],
                    &[0.0, 0.0, 0.0]],
                  &[true, true, false, false],
                  Some(print_callback)) <= 0.01);
}
