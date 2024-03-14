use plotters::prelude::*;
use std::ops::Range;

fn visualize_amdahls_law() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("amdahls_law.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Amdahl's Law", ("Arial", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..1f32, 0f32..10f32)?;

    chart.configure_mesh().draw()?;

    let n = 4; // number of processors
    let p_range: Range<f32> = Range { start: 0.01, end: 1.0 }; // proportion of the program that can be parallelized
    let p_values: Vec<f32> = (p_range.start..p_range.end).step_by(0.01).collect();
    let speedup_values: Vec<f32> = p_values.iter().map(|&p| 1.0 / ((1.0 - p) + p / n as f32)).collect();

    chart.draw_series(LineSeries::new(
        p_values.iter().zip(speedup_values.iter()).map(|(x, y)| (*x, *y)),
        &RED,
    ))?;

    Ok(())
}