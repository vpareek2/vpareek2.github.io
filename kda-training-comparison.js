(function () {
  const figure = document.querySelector('[data-training-comparison]');
  if (!figure) return;

  const traces = {
    eager: [[0,10.39512,1],[1.6887,10.39412,2],[3.3813,10.39237,3],[5.0806,10.35663,4],[6.7833,10.28358,5],[8.4858,10.17066,6],[10.1893,9.98209,7],[11.8913,10.23951,8],[13.5927,9.93692,9],[15.294,9.68773,10],[16.9953,9.47413,11],[18.697,9.26603,12],[20.3969,9.09164,13]],
    fla: [[0,10.39512,1],[.0124,10.39412,2],[.5066,7.82376,42],[1.0018,6.52465,82],[1.4974,6.06932,122],[1.9933,5.76646,162],[2.4885,5.50834,202],[2.9824,5.28661,242],[3.4767,5.19195,282],[3.9722,4.9707,322],[4.4673,5.02212,362],[4.9626,4.80408,402],[5.4576,4.82859,442],[5.9518,4.70933,482],[6.4465,4.60702,522],[6.9417,4.62812,562],[7.4374,4.56625,602],[7.9326,4.43398,642],[8.4272,4.42929,682],[8.921,4.3234,722],[9.4159,4.41447,762],[9.911,4.30972,802],[10.4062,4.26543,842],[10.9006,4.26407,882],[11.3956,4.18515,922],[11.8903,4.16688,962],[12.385,4.14348,1002],[12.879,4.10354,1042],[13.3737,4.06217,1082],[13.8692,4.15691,1122],[14.3646,4.03078,1162],[14.8589,3.96009,1202],[15.3541,4.03819,1242],[15.8498,3.97343,1282],[16.345,3.96867,1322],[16.8391,3.93963,1362],[17.3328,3.93613,1402],[17.8277,3.93853,1442],[18.3237,3.90557,1482],[18.818,3.89304,1522],[19.3121,3.86789,1562],[19.8075,3.86775,1602],[20.0054,3.85048,1618]],
    project: [[0,10.39512,1],[.0121,10.39412,2],[.4961,7.94978,42],[.981,6.54624,82],[1.4658,6.08853,122],[1.9525,5.79247,162],[2.4477,5.53028,202],[2.9418,5.30167,242],[3.4291,5.20485,282],[3.9141,4.97963,322],[4.3994,5.03233,362],[4.8849,4.81318,402],[5.3704,4.83912,442],[5.8554,4.71967,482],[6.341,4.61607,522],[6.8258,4.64212,562],[7.3161,4.58077,602],[7.8087,4.44711,642],[8.2933,4.4434,682],[8.7775,4.34061,722],[9.2617,4.423,762],[9.7461,4.31968,802],[10.2307,4.27279,842],[10.7148,4.27091,882],[11.1992,4.19248,922],[11.6833,4.1692,962],[12.1682,4.14853,1002],[12.6525,4.11034,1042],[13.1371,4.07022,1082],[13.6218,4.16163,1122],[14.1062,4.03762,1162],[14.5909,3.96445,1202],[15.0763,4.03852,1242],[15.5705,3.97654,1282],[16.0642,3.96778,1322],[16.5579,3.9364,1362],[17.0515,3.93522,1402],[17.5453,3.93644,1442],[18.0389,3.90597,1482],[18.5323,3.88835,1522],[19.0257,3.86569,1562],[19.5194,3.86561,1602],[20.0122,3.81096,1642]]
  };

  const ns = 'http://www.w3.org/2000/svg';
  const width = 360;
  const height = 230;
  const margin = { top: 14, right: 13, bottom: 29, left: 35 };
  const xMax = 20.5;
  const yMin = 3.5;
  const yMax = 10.6;

  const x = (value) => margin.left + (value / xMax) * (width - margin.left - margin.right);
  const y = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * (height - margin.top - margin.bottom);

  function node(name, attributes, text) {
    const element = document.createElementNS(ns, name);
    Object.entries(attributes || {}).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function formatTime(minutes) {
    const totalSeconds = Math.max(0, Math.round(minutes * 60));
    return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, '0')}`;
  }

  function render(card) {
    const name = card.dataset.trainingRun;
    const points = traces[name];
    const svg = card.querySelector('svg');
    const tooltip = card.querySelector('.training-chart-tooltip');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

    [4, 6, 8, 10].forEach((tick) => {
      svg.append(node('line', { class: 'training-grid-line', x1: margin.left, x2: width - margin.right, y1: y(tick), y2: y(tick) }));
      svg.append(node('text', { class: 'training-axis-tick training-axis-y', x: margin.left - 8, y: y(tick) + 3 }, tick));
    });

    [0, 10, 20].forEach((tick) => {
      svg.append(node('line', { class: 'training-grid-mark', x1: x(tick), x2: x(tick), y1: margin.top, y2: height - margin.bottom }));
      svg.append(node('text', { class: 'training-axis-tick training-axis-x', x: x(tick), y: height - 8 }, tick === 20 ? '20m' : tick));
    });

    const pathData = points.map((point, index) => `${index ? 'L' : 'M'} ${x(point[0]).toFixed(2)} ${y(point[1]).toFixed(2)}`).join(' ');
    svg.append(node('path', { class: 'training-loss-line', d: pathData, pathLength: 1 }));

    const finalPoint = points[points.length - 1];
    svg.append(node('circle', { class: 'training-endpoint-halo', cx: x(finalPoint[0]), cy: y(finalPoint[1]), r: 7 }));
    svg.append(node('circle', { class: 'training-endpoint', cx: x(finalPoint[0]), cy: y(finalPoint[1]), r: 3.2 }));

    const focus = node('circle', { class: 'training-focus-point', r: 4, hidden: '' });
    const crosshair = node('line', { class: 'training-crosshair', y1: margin.top, y2: height - margin.bottom, hidden: '' });
    svg.append(crosshair, focus);

    const hitArea = node('rect', {
      class: 'training-hit-area',
      x: margin.left,
      y: margin.top,
      width: width - margin.left - margin.right,
      height: height - margin.top - margin.bottom,
      tabindex: '0',
      role: 'button',
      'aria-label': `Inspect ${name} training trace`
    });
    svg.append(hitArea);

    function show(point) {
      focus.removeAttribute('hidden');
      crosshair.removeAttribute('hidden');
      focus.setAttribute('cx', x(point[0]));
      focus.setAttribute('cy', y(point[1]));
      crosshair.setAttribute('x1', x(point[0]));
      crosshair.setAttribute('x2', x(point[0]));
      tooltip.hidden = false;
      tooltip.innerHTML = `<span>${formatTime(point[0])}</span><b>UPDATE ${Math.max(0, point[2] - 1).toLocaleString()}</b><strong>LOSS ${point[1].toFixed(3)}</strong>`;
      const left = Math.min(74, Math.max(26, (x(point[0]) / width) * 100));
      tooltip.style.left = `${left}%`;
    }

    function hide() {
      focus.setAttribute('hidden', '');
      crosshair.setAttribute('hidden', '');
      tooltip.hidden = true;
    }

    hitArea.addEventListener('pointermove', (event) => {
      const rect = svg.getBoundingClientRect();
      const chartX = ((event.clientX - rect.left) / rect.width) * width;
      const minute = Math.max(0, Math.min(xMax, ((chartX - margin.left) / (width - margin.left - margin.right)) * xMax));
      show(points.reduce((best, point) => Math.abs(point[0] - minute) < Math.abs(best[0] - minute) ? point : best));
    });
    hitArea.addEventListener('pointerleave', hide);
    hitArea.addEventListener('focus', () => show(finalPoint));
    hitArea.addEventListener('blur', hide);
  }

  figure.querySelectorAll('[data-training-run]').forEach(render);
  figure.classList.add('is-prepared');

  if (!('IntersectionObserver' in window)) {
    figure.classList.add('is-visible');
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    if (!entries.some((entry) => entry.isIntersecting)) return;
    figure.classList.add('is-visible');
    observer.disconnect();
  }, { threshold: 0.15 });
  observer.observe(figure);
})();
