```{=html}
<div id="interference-plot"></div>
```

```{=html}
<div class="slider-container">
    Phase difference: <input type="range" min="0" max="360" value="0" id="phaseSlider">
    <span id="phaseValue">0°</span>
</div>

<style>
    .line {
        fill: none;
        stroke-width: 2;
    }
    .slider-container {
        margin: 20px;
    }
    .axis-label {
        font-family: sans-serif;
        font-size: 12px;
    }
</style>

<script src="https://d3js.org/d3.v7.min.js"></script>

<script>
    // Set up dimensions
    const width = 600;
    const height = 300;
    const margin = {top: 20, right: 20, bottom: 30, left: 40};

    // Create SVG
    const svg = d3.select("#interference-plot")
        .append("svg")
        .attr("width", width)
        .attr("height", height);

    // Set up scales
    const x = d3.scaleLinear()
        .domain([0, 6 * Math.PI])
        .range([margin.left, width - margin.right]);

    const y = d3.scaleLinear()
        .domain([-2.5, 2.5])
        .range([height - margin.bottom, margin.top]);

    // Add axes
    svg.append("g")
        .attr("transform", `translate(0,${height/2})`)
        .call(d3.axisBottom(x));

    svg.append("g")
        .attr("transform", `translate(${margin.left},0)`)
        .call(d3.axisLeft(y));

    svg.append("text")
        .attr("class", "axis-label")
        .attr("text-anchor", "middle")
        .attr("x", width/2)
        .attr("y", height - 10)
        .text("Phase (radians)");

    svg.append("text")
        .attr("class", "axis-label")
        .attr("text-anchor", "middle")
        .attr("transform", "rotate(-90)")
        .attr("x", height/2)
        .attr("y", 20)
        .text("Amplitude");


    // Line generator
    const line = d3.line()
        .x(d => x(d.x))
        .y(d => y(d.y));

    // Generate data points
    function generateData(phaseShift) {
        const points = [];
        for (let i = 0; i <= 100; i++) {
            const x = (i/100) * 6 * Math.PI;
            points.push({
                x: x,
                y1: Math.sin(x),
                y2: Math.sin(x + phaseShift * Math.PI/180),
                sum: Math.sin(x) + Math.sin(x - phaseShift * Math.PI/180)
            });
        }
        return points;
    }

    // Initial data
    let data = generateData(0);

    // Draw lines
    const wave1 = svg.append("path")
        .attr("class", "line")
        .style("stroke", "blue");

    const wave2 = svg.append("path")
        .attr("class", "line")
        .style("stroke", "red");

    const sumWave = svg.append("path")
        .attr("class", "line")
        .style("stroke", "purple");

    // Update function
    function update(phaseShift) {
        data = generateData(phaseShift);

        wave1.attr("d", line(data.map(d => ({x: d.x, y: d.y1}))));
        wave2.attr("d", line(data.map(d => ({x: d.x, y: d.y2}))));
        sumWave.attr("d", line(data.map(d => ({x: d.x, y: d.sum}))));
    }

    // Add slider functionality
    d3.select("#phaseSlider").on("input", function() {
        const phase = +this.value;
        d3.select("#phaseValue").text(phase + "°");
        update(phase);
    });

    // Initial update
    update(0);

    // Add legend
    const legend = svg.append("g")
        .attr("font-family", "sans-serif")
        .attr("font-size", 12)
        .attr("transform", `translate(${width - 100},${margin.top})`);

    legend.append("rect")
        .attr("x", 0)
        .attr("width", 19)
        .attr("height", 19)
        .attr("fill", "blue");
    legend.append("text")
        .attr("x", 24)
        .attr("y", 9.5)
        .attr("dy", "0.32em")
        .text("wave 1");

    legend.append("rect")
        .attr("x", 0)
        .attr("y", 20)
        .attr("width", 19)
        .attr("height", 19)
        .attr("fill", "red");
    legend.append("text")
        .attr("x", 24)
        .attr("y", 29.5)
        .attr("dy", "0.32em")
        .text("wave 2");

    legend.append("rect")
        .attr("x", 0)
        .attr("y", 40)
        .attr("width", 19)
        .attr("height", 19)
        .attr("fill", "purple");
    legend.append("text")
        .attr("x", 24)
        .attr("y", 49.5)
        .attr("dy", "0.32em")
        .text("sum");

</script>
```
