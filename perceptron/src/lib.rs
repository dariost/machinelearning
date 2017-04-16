pub struct Perceptron
{
    weight: Vec<f64>,
    learning_rate: f64,
    iterations: usize,
}

impl Perceptron
{
    pub fn new(num_features: usize, learning_rate: f64, iterations: usize) -> Perceptron
    {
        Perceptron {
            weight: vec![0.0; num_features + 1],
            learning_rate: learning_rate,
            iterations: iterations,
        }
    }

    #[cfg_attr(rustfmt, rustfmt_skip)]
    pub fn fit(&mut self, data: &[&[f64]], class: &[bool]) -> usize
    {
        let class: Vec<f64> = class.iter().map(|&x| if x {1.0} else {-1.0}).collect();
        assert_eq!(data.len(), class.len());
        assert!(data.iter().all(|x| x.len() == self.weight.len() - 1));
        let mut errors = 0;
        for it in 0..self.iterations
        {
            errors = 0;
            for i in 0..data.len()
            {
                let prediction = if self.predict(data[i]) {1.0} else {-1.0};
                let update = self.learning_rate * (class[i] - prediction);
                if update != 0.0
                {
                    self.weight[0] += update;
                    for j in 1..self.weight.len()
                    {
                        self.weight[j] += update * data[i][j - 1];
                    }
                    errors += 1;
                }
            }
            println!("Iteration #{}: {} error(s)", it, errors);
            println!("Weight: {:?}", self.weight);
            if errors == 0
            {
                break;
            }
        }
        errors
    }

    pub fn predict(&self, data: &[f64]) -> bool
    {
        assert_eq!(data.len(), self.weight.len() - 1);
        let value = self.weight
            .iter()
            .skip(1)
            .zip(data.iter())
            .map(|x| x.0 * x.1)
            .sum::<f64>() + self.weight[0];
        value >= 0.0
    }
}

#[test]
fn trivial()
{
    let mut p = Perceptron::new(2, 0.1, 100);
    assert_eq!(p.fit(&[&[1.0, 0.0], &[4.0, 1.0], &[0.0, 1.0], &[2.0, 3.0]], &[true, false, true, false]), 0);
    assert_eq!(p.predict(&[-1.0, -1.0]), true);
    assert_eq!(p.predict(&[10.0, 10.0]), false);
    assert_eq!(p.predict(&[0.5, 0.5]), true);
}

#[test]
#[should_panic]
fn trivial_wrong()
{
    let mut p = Perceptron::new(2, 0.1, 100);
    assert_eq!(p.fit(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[0.0, 0.0]], &[true, true, false, false]), 0);
}

#[test]
fn trivial_3d()
{
    let mut p = Perceptron::new(3, 0.1, 100);
    assert_eq!(p.fit(&[&[1.0, 0.0, 1.0],
                       &[0.0, 1.0, 1.0],
                       &[1.0, 1.0, 0.0],
                       &[0.0, 0.0, 0.0]],
                     &[true, true, false, false]),
               0);
}
