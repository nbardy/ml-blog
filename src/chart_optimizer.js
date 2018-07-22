import * as c from 'chart.js'

const rate = 1;
// Attaches to minimize
export function trackOptimizer(optimizer, canvas ) {
  const ctx = canvas.getContext('2d');
  const lossData = [];
  const entropyData = [];
  const config = {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: "Loss",
          yAxisID: 'A',
          pointBackgroundColor: '#F412A1',
          data: lossData
        },
        {
          label: "Entropy",
          yAxisID: 'B',
          pointBackgroundColor: '#14D201',
          data: entropyData
        }
      ]
    },
    options: {
      responsive: false,
      scales: {
        yAxes: [
          {
            id: 'A',
            type: 'linear',
            position: 'left'
          },
          {
            id: 'B',
            type: 'linear',
            position: 'left'
          }
        ]
      }
    }
  }

  const chart = new c.Chart(ctx, config);

  var i = 0;

  if((typeof optimizer.minimize) == 'function') {
    const oldF = optimizer.minimize;
    optimizer.minimize = function(f) {
      // TODO: Make tensor based graph for performance
      i++;
      const loss = oldF.call(this, f).dataSync()[0]
      if((i % rate ) == 0) {
        lossData.push({x: i, y: loss});
        entropyData.push({x: i, y: this.entropy});
        chart.update()
      }
    }

  } else {
    console.error("Optimizer must have a minimize function to wrap");
    throw new Error("Missing minimize");
  }
}
