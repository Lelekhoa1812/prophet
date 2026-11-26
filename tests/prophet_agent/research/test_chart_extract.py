"""Chart series read from page markup: exact values for each library and for an SVG line chart with known geometry."""
import math

import pytest

from prophet_agent.research.charts import EMBEDDED, SVG_GEOMETRY, LiteralReader, extract_page_series, filter_series


def _page(*scripts: str, body: str = "") -> str:
    tags = "".join(f"<script>\n{s}\n</script>" for s in scripts)
    return f"<html><head>{tags}</head><body>{body}</body></html>"


def _by_name(ex, name):
    matches = [s for s in ex.series if s["name"] == name]
    assert len(matches) == 1, (name, [s["name"] for s in ex.series], ex.skipped)
    return matches[0]


def test_highcharts_categories_and_numbers_exact():
    html = _page("""
        Highcharts.chart('c', {
          chart: { type: 'line' },   // a comment
          xAxis: { categories: ['Jan', 'Feb', 'Mar'], },
          yAxis: { labels: { formatter: function () { return this.value + ' kg'; } } },
          series: [{ name: 'Sales', data: [1.5, 2, 3.25] }],
        });
    """)
    ex = extract_page_series(html)
    s = _by_name(ex, "Sales")
    assert s["method"] == "highcharts"
    assert s["exactness"] == EMBEDDED
    assert s["x"] == ["Jan", "Feb", "Mar"]
    assert s["y"] == [1.5, 2, 3.25]
    assert s["x_kind"] == "category"
    assert s["source"]["kind"] == "script" and s["source"]["call"].startswith("Highcharts.chart")


def test_highcharts_datetime_pairs_use_date_utc_and_pie_points():
    html = _page("""
        var opts = { xAxis: { type: 'datetime' },
          series: [{ name: 'Daily', data: [[Date.UTC(2024, 0, 1), 10], [Date.UTC(2024, 0, 2), 12.5]] },
                   { name: 'Share', type: 'pie', data: [{ name: 'A', y: 60 }, { name: 'B', y: 40 }] }] };
        Highcharts.stockChart('d', opts);
    """)
    ex = extract_page_series(html)
    daily = _by_name(ex, "Daily")
    assert daily["x"] == ["2024-01-01", "2024-01-02"]
    assert daily["y"] == [10, 12.5]
    assert daily["x_kind"] == "datetime"
    share = _by_name(ex, "Share")
    assert share["x"] == ["A", "B"] and share["y"] == [60, 40]


def test_plotly_traces_from_a_variable_exact():
    html = _page("""
        var xs = [1, 2, 3];
        var trace = { x: xs, y: [4, 5, 6.5], name: 'Price', type: 'scatter' };
        Plotly.newPlot('plot', [trace], { title: 'Price' }, { responsive: true });
    """)
    ex = extract_page_series(html)
    s = _by_name(ex, "Price")
    assert s["method"] == "plotly"
    assert s["x"] == [1, 2, 3] and s["y"] == [4, 5, 6.5]


def test_chartjs_labels_and_datasets_exact():
    html = _page("""
        const config = { type: 'bar', data: { labels: ['Q1', 'Q2'],
          datasets: [{ label: 'Revenue', data: [10, 20] }, { label: 'Cost', data: [4, 8] }] },
          options: { scales: { y: { ticks: { callback: (v) => v + '%' } } } } };
        new Chart(document.getElementById('x'), config);
    """)
    ex = extract_page_series(html)
    rev = _by_name(ex, "Revenue")
    assert rev["method"] == "chartjs"
    assert rev["x"] == ["Q1", "Q2"] and rev["y"] == [10, 20]
    assert _by_name(ex, "Cost")["y"] == [4, 8]


def test_echarts_set_option_category_and_pairs_exact():
    html = _page("""
        chart.setOption({
          xAxis: { type: 'category', data: ['Mon', 'Tue', 'Wed'] },
          yAxis: {},
          series: [{ name: 'Visits', type: 'line', data: [5, 7, 9] },
                   { name: 'Pairs', type: 'scatter', data: [[1, 2], [3, 4]] }],
        });
    """)
    ex = extract_page_series(html)
    v = _by_name(ex, "Visits")
    assert v["method"] == "echarts"
    assert v["x"] == ["Mon", "Tue", "Wed"] and v["y"] == [5, 7, 9]
    p = _by_name(ex, "Pairs")
    assert p["x"] == [1, 3] and p["y"] == [2, 4]


def test_vega_lite_inline_values_grouped_by_colour():
    html = _page("""
        vegaEmbed('#v', {
          $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
          data: { values: [{ d: 'a', v: 1, g: 'x' }, { d: 'b', v: 2, g: 'x' }, { d: 'a', v: 3, g: 'y' }] },
          mark: 'line',
          encoding: { x: { field: 'd', type: 'nominal' }, y: { field: 'v', type: 'quantitative' },
                      color: { field: 'g' } }
        });
    """)
    ex = extract_page_series(html)
    x = _by_name(ex, "x")
    assert x["method"] == "vega" and x["x"] == ["a", "b"] and x["y"] == [1, 2]
    assert _by_name(ex, "y")["y"] == [3]


def test_vega_lite_aggregate_is_not_reported_as_embedded_value():
    html = _page("""
        vegaEmbed('#v', { data: { values: [{ d: 'a', v: 1 }] }, mark: 'bar',
          encoding: { x: { field: 'd' }, y: { field: 'v', aggregate: 'sum' } } });
    """)
    ex = extract_page_series(html)
    assert ex.series == []
    assert any("aggregated" in s for s in ex.skipped)


def test_unreadable_series_is_skipped_not_guessed():
    html = _page("""
        Highcharts.chart('c', { series: [{ name: 'Good', data: [1, 2] },
           { name: 'Bad', data: [1, Math.random(), 3] }] });
    """)
    ex = extract_page_series(html)
    assert [s["name"] for s in ex.series] == ["Good"]
    assert any("Bad" in s for s in ex.skipped)


# -- SVG line chart with known geometry ---------------------------------------------------------------------------------
# value x -> pixel 50 + 20*x (ticks 0, 5, 10 at 50, 150, 250); value y -> pixel 200 - 10*y (ticks 0, 10, 20 at 200, 100, 0)

SVG_LINE = """
<div><svg width="300" height="220" viewBox="0 0 300 220">
  <g transform="translate(10, 0)">
    <text x="40" y="200">0</text><text x="40" y="100">10</text><text x="40" y="0">20</text>
    <text x="50" y="215">0</text><text x="150" y="215">5</text><text x="250" y="215">10</text>
    <path d="M50 100 L250 100" stroke="#ccc"/>
    <polyline id="sales" points="50,200 150,100 250,0" stroke="#d62728" fill="none"><title>Sales</title></polyline>
  </g>
</svg></div>
"""


def test_svg_line_chart_inverts_axis_ticks_exactly():
    ex = extract_page_series(_page(body=SVG_LINE))
    s = _by_name(ex, "Sales")
    assert s["method"] == "svg"
    assert s["exactness"] == SVG_GEOMETRY
    assert [round(v, 9) for v in s["x"]] == [0.0, 5.0, 10.0]
    assert [round(v, 9) for v in s["y"]] == [0.0, 10.0, 20.0]
    assert all(math.isfinite(v) for v in s["y"])
    # The two-point gridline is not a series.
    assert len(ex.series) == 1


def test_svg_with_one_tick_per_axis_is_not_read():
    svg = ('<svg><text x="40" y="200">0</text><polyline points="50,200 150,100 250,0"/></svg>')
    ex = extract_page_series(_page(body=svg))
    assert ex.series == []


def test_svg_inside_a_library_container_is_left_to_the_script():
    html = _page(body='<div class="highcharts-container">' + SVG_LINE + '</div>')
    ex = extract_page_series(html)
    assert ex.series == []


def test_literal_reader_rejects_functions_and_keeps_siblings():
    value, _ = LiteralReader("{a: function (x) { return x, 2; }, b: [1, 'two', `three`], c: null,}").value(0)
    assert value["a"] is not None and type(value["a"]).__name__ == "_Unparsed"
    assert value["b"] == [1, "two", "three"] and value["c"] is None


def test_filter_series_by_name():
    series = [{"name": "Gross margin"}, {"name": "Net sales"}]
    assert [s["name"] for s in filter_series(series, "margin")] == ["Gross margin"]
    assert len(filter_series(series, None)) == 2


@pytest.mark.parametrize("text", ["{a: 1", "[1, 2", "'unterminated"])
def test_broken_literals_raise_nothing_out_of_extract(text):
    ex = extract_page_series(_page(f"Highcharts.chart('c', {text});"))
    assert ex.series == []
