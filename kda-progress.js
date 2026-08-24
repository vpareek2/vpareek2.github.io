(function () {
  var root = document.querySelector('[data-kda-chart]');
  if (!root) return;

  var data = [
    [0,833.5,'Initial eager implementation','milestone','initial-baseline'],
    [14,1978,'Parallel backward history','kept','speed-attempt-14'],
    [15,2866,'Parallel forward history','kept','speed-attempt-15'],
    [18,6689,'Value-tiled reverse','kept','speed-attempt-18'],
    [19,7394,'First practical project CUDA baseline','milestone','speed-attempt-19'],
    [23,10957,'Convolution bounded dependencies','kept','attempt-00023'],
    [24,12977,'Factored convolution gradient','kept','attempt-00024'],
    [26,14800,'FP32 C64 BMM scan','kept','attempt-00026'],
    [27,14981,'BF16 WMMA C64 scan','discarded','attempt-00027'],
    [28,15895,'Chunk-boundary recompute','kept','attempt-00028'],
    [34,19009,'Local raw-gate gradient','kept','attempt-00034'],
    [36,21847,'Row-parallel pair VJP','kept','attempt-00036'],
    [38,23851,'Parallel exact beta gradient','kept','attempt-00038'],
    [43,24310,'Fused stable transforms','kept','attempt-00043'],
    [45,25420,'Key-major parameter reduction','kept','attempt-00045'],
    [46,26258,'Boundary key block','kept','attempt-00046'],
    [47,26685,'Finalize row block','discarded','attempt-00047'],
    [51,27043,'Bounded stable pair VJP','kept','attempt-00051'],
    [53,27671,'Chunk partial finalization','kept','attempt-00053'],
    [65,28325,'Tensor-core recurrence','milestone','attempt-00065'],
    [67,28663,'Pair pack 256','discarded','attempt-00067'],
    [68,28784,'Eight-warp forward','kept','attempt-00068'],
    [70,29699,'Eight-warp group recompute','kept','attempt-00070'],
    [72,29868,'Value-tile group state','kept','attempt-00072'],
    [77,29921,'Reverse transfer fusion','kept','attempt-00077'],
    [83,30637,'Pair WMMA VJP','kept','attempt-00083'],
    [84,31747,'Build pair WMMA','kept','attempt-00084'],
    [86,31981,'Persistent build solve','discarded','attempt-00086'],
    [89,31856,'BF16 inter-chunk state','discarded','attempt-00089'],
    [91,32914,'Persistent reverse group','kept','attempt-00091'],
    [93,33226,'Fused group producer','discarded','attempt-00093'],
    [95,33348,'Post-reverse WMMA VJP','discarded','attempt-00095'],
    [97,33236,'Two-CTA post-reverse VJP','discarded','attempt-00097'],
    [100,33601,'Colored pair VJP','kept','attempt-00100'],
    [110,34210,'Backward group-major layout','discarded','attempt-00110'],
    [111,34029,'Group-major producer','discarded','attempt-00111'],
    [118,33689,'Forward tile prefetch','discarded','attempt-00118'],
    [123,29354,'BF16 chunk-state history','discarded','attempt-00123'],
    [125,34413,'Async scan operands','discarded','attempt-00125'],
    [127,34494,'Preprocess q-gamma','kept','attempt-00127'],
    [133,34926,'Two-group short-path guard','discarded','attempt-00133'],
    [134,34549,'Boundary decay cache','discarded','attempt-00134'],
    [154,34468,'Fused state dot','discarded','attempt-00154'],
    [156,34672,'Fused BF16 reverse products','discarded','attempt-00156'],
    [161,35521,'Fast math and generic fallback','kept','attempt-00161'],
    [162,35984,'Four-warp VJP','discarded','attempt-00162'],
    [165,35743,'Fused U/W pack','discarded','attempt-00165'],
    [168,36185,'Flattened parallel backward','milestone','attempt-00168'],
    [173,35981,'Four-CTA VJP','discarded','attempt-00173'],
    [175,36719.5,'Tiled convolution backward','kept','attempt-00175'],
    [189,37198,'Register dP consumer','discarded','attempt-00189'],
    [190,37519,'Register dQ consumer','discarded','attempt-00190'],
    [194,37701,'Register state products','discarded','attempt-00194'],
    [201,38052,'Direct WMMA dH scan','discarded','attempt-00201'],
    [204,38803,'Forward group checkpoints','kept','attempt-00204'],
    [211,39560,'GB10 register forward state','discarded','attempt-00211'],
    [213,40076,'Retained forward WY factors','kept','attempt-00213'],
    [217,40347,'Correct retained WY layout','kept','attempt-00217'],
    [221,39784,'Persistent build solve','discarded','attempt-00221'],
    [222,40632,'Pipelined key products','discarded','attempt-00222'],
    [227,40707,'Split fused boundary dH','discarded','attempt-00227'],
    [231,40834,'Hidden dA tail','kept','attempt-00231'],
    [243,41565,'Fused retention scan pack','discarded','attempt-00243'],
    [244,41523,'Fused reverse base gradient','discarded','attempt-00244'],
    [245,41657,'Retained warp normalization','kept','attempt-00245'],
    [255,41922,'Swapped retained P for prefix','kept','attempt-00255'],
    [256,42109,'BF16 forward WY products','kept','attempt-00256'],
    [265,41856,'Direct publication register VJP','discarded','attempt-00265'],
    [266,42237,'Compact BF16 dataflow','milestone','attempt-00266'],
    [268,42199,'BF16 U/W register VJP','discarded','attempt-00268'],
    [270,41975,'Fused qg/kg producer','discarded','attempt-00270'],
    [271,42262,'Forward CUDA Graph','discarded','attempt-00271'],
    [272,40023,'Reverse-group CUDA Graph','discarded','attempt-00272'],
    [274,20883,'Stacked CUDA Graphs','discarded','attempt-00274'],
    [279,41249,'Interleaved group pack','discarded','attempt-00279'],
    [282,41530,'Restored-k interleaved stack','discarded','attempt-00282'],
    [283,41132,'Fused preprocess/build stack','discarded','attempt-00283'],
    [285,42064,'BF16 group-U rebuild','discarded','attempt-00285'],
    [289,42721,'Backward zero-fill producers','discarded','attempt-00289'],
    [292,41680,'Optimized host wrapper','discarded','attempt-00292'],
    [308,43145,'Preprocess and convolution stack','discarded','attempt-00308'],
    [321,43260,'FP16 normalized forward scratch','discarded','attempt-00321'],
    [325,43572,'Fused best stack','discarded','attempt-00325'],
    [335,43572,'Compact retained q/k/P','discarded','attempt-00335'],
    [342,43840,'Group-major retained data','kept','attempt-00342'],
    [366,44942,'Exact release confirmation','milestone','release-confirmation']
  ].map(function (row) {
    return { x: row[0], tps: row[1], label: row[2], status: row[3], id: row[4] };
  });

  var milestones = data.filter(function (point) { return point.status === 'milestone'; });
  var retainedPath = data.filter(function (point) { return point.status !== 'discarded'; });
  var svg = root.querySelector('svg');
  var tooltip = root.querySelector('.campaign-tooltip');
  var railList = document.querySelector('.campaign-rail-list');
  var namespace = 'http://www.w3.org/2000/svg';
  var flaTps = 43937;
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var activePoint = milestones[0];
  var milestoneDetails = {
    'initial-baseline': 'Python-loop baseline: 38.2 seconds per training update.',
    'speed-attempt-19': 'First practical fully owned CUDA training path.',
    'attempt-00065': 'A unified WMMA scan moved recurrence onto tensor cores.',
    'attempt-00168': 'A flatter pair grid removed 18 kernel launches.',
    'attempt-00266': 'BF16 publication removed large FP32 surfaces.',
    'release-confirmation': 'Three-run median; pinned FLA reached 43,937.'
  };

  function make(name, attributes, text) {
    var node = document.createElementNS(namespace, name);
    Object.keys(attributes || {}).forEach(function (key) {
      node.setAttribute(key, attributes[key]);
    });
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function formatTps(value) {
    return Math.round(value).toLocaleString('en-US');
  }

  function formatMilestoneTps(point) {
    return point.id === 'initial-baseline' ? '833' : formatTps(point.tps);
  }

  function statusLabel(point) {
    if (point.status === 'milestone') return 'Highlighted milestone';
    if (point.status === 'kept') return 'Retained exact improvement';
    return 'Measured, not retained';
  }

  function pointDetail(point) {
    return milestoneDetails[point.id] || '';
  }

  function setActivePoint(point, shouldScroll) {
    var activeCard;
    activePoint = point;
    railList.querySelectorAll('.campaign-rail-card').forEach(function (card) {
      var isActive = card.dataset.pointId === point.id;
      card.classList.toggle('is-active', isActive);
      if (isActive) {
        card.setAttribute('aria-current', 'step');
        activeCard = card;
      } else {
        card.removeAttribute('aria-current');
      }
    });

    svg.querySelectorAll('.chart-point').forEach(function (circle) {
      circle.classList.toggle('is-active', circle.dataset.pointId === point.id);
    });
    svg.querySelectorAll('.chart-milestone-target').forEach(function (label) {
      label.classList.toggle('is-active', label.dataset.pointId === point.id);
    });

    if (activeCard && shouldScroll) {
      var listRect = railList.getBoundingClientRect();
      var cardRect = activeCard.getBoundingClientRect();
      var target = railList.scrollTop + cardRect.top - listRect.top;
      railList.scrollTo({ top: Math.max(0, target), behavior: reduceMotion ? 'auto' : 'smooth' });
    }
  }

  function selectManually(point) {
    setActivePoint(point, true);
  }

  function showTooltip(point, cx, cy) {
    var detail = pointDetail(point);
    tooltip.innerHTML = '<strong>' + point.label + '</strong>' +
      (detail ? '<small>' + detail + '</small>' : '') + '<span>' +
      formatTps(point.tps) + ' tokens/s · ' + statusLabel(point) + '</span>';
    tooltip.hidden = false;
    var tooltipWidth = tooltip.offsetWidth;
    var tooltipHeight = tooltip.offsetHeight;
    var left = Math.max(8, Math.min(root.clientWidth - tooltipWidth - 8, cx + 12));
    var top = Math.max(8, cy - tooltipHeight - 10);
    tooltip.style.left = left + 'px';
    tooltip.style.top = top + 'px';
  }

  function hideTooltip() {
    tooltip.hidden = true;
  }

  function draw() {
    var width = Math.max(300, Math.round(root.clientWidth));
    var compact = width < 560;
    var height = compact ? 350 : 440;
    var margin = { top: 35, right: 14, bottom: 36, left: compact ? 42 : 52 };
    var innerWidth = width - margin.left - margin.right;
    var innerHeight = height - margin.top - margin.bottom;
    var yMax = 48000;

    function xScale(value) {
      return margin.left + Math.sqrt(value / 366) * innerWidth;
    }

    function yScale(value) {
      return margin.top + innerHeight - (value / yMax) * innerHeight;
    }

    svg.replaceChildren();
    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);

    [0,10000,20000,30000,40000].forEach(function (tick) {
      var y = yScale(tick);
      svg.appendChild(make('line', { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: 'chart-grid' }));
      svg.appendChild(make('text', { x: margin.left - 8, y: y + 3, 'text-anchor': 'end', class: 'chart-tick' }, tick === 0 ? '0' : (tick / 1000) + 'K'));
    });

    svg.appendChild(make('text', {
      x: margin.left,
      y: height - 7,
      class: 'chart-axis-label'
    }, 'CAMPAIGN PROGRESS →'));

    var flaY = yScale(flaTps);
    svg.appendChild(make('line', { x1: margin.left, y1: flaY, x2: width - margin.right, y2: flaY, class: 'chart-reference' }));
    svg.appendChild(make('text', {
      x: width - margin.right,
      y: flaY - 7,
      'text-anchor': 'end',
      class: 'chart-reference-label'
    }, compact ? 'FLA 43,937' : 'FLA RELEASE COMPARATOR · 43,937 TOKENS/S'));

    var story = retainedPath.map(function (point) {
      return xScale(point.x) + ',' + yScale(point.tps);
    }).join(' ');
    var storyLine = make('polyline', { points: story, class: 'chart-story chart-story-animated' });
    svg.appendChild(storyLine);
    var pathLength = storyLine.getTotalLength();
    storyLine.style.setProperty('--chart-path-length', pathLength);
    storyLine.style.strokeDasharray = pathLength;

    data.forEach(function (point) {
      var cx = xScale(point.x);
      var cy = yScale(point.tps);
      var radius = point.status === 'milestone' ? (compact ? 5.4 : 6.2) : 2.5;
      var pointDelay = 1.15 + Math.sqrt(point.x / 366) * 1.95;
      var circle = make('circle', {
        cx: cx,
        cy: cy,
        r: radius,
        class: 'chart-point chart-point-' + point.status + ' chart-point-animated',
        'data-point-id': point.id,
        tabindex: '0',
        role: 'button',
        'aria-label': point.label + ', ' + formatTps(point.tps) + ' tokens per second, ' + statusLabel(point)
      });
      circle.style.animationDelay = pointDelay + 's';
      circle.addEventListener('pointerenter', function () {
        showTooltip(point, cx, cy);
      });
      circle.addEventListener('pointerleave', hideTooltip);
      circle.addEventListener('focus', function () {
        showTooltip(point, cx, cy);
      });
      circle.addEventListener('blur', hideTooltip);
      circle.addEventListener('click', function () { selectManually(point); });
      svg.appendChild(circle);
    });

    var milestoneLabelOffsets = {
      'initial-baseline': [14, -25],
      'speed-attempt-19': [16, 29],
      'attempt-00065': [-16, -28],
      'attempt-00168': [-16, -30],
      'attempt-00266': [16, 46],
      'release-confirmation': [-16, -28]
    };
    var milestoneTitles = {
      'initial-baseline': 'INITIAL EAGER',
      'speed-attempt-19': 'PROJECT CUDA BASELINE',
      'attempt-00065': 'TENSOR-CORE RECURRENCE',
      'attempt-00168': 'FLATTENED BACKWARD',
      'attempt-00266': 'COMPACT BF16 DATAFLOW',
      'release-confirmation': 'RELEASE CONFIRMATION'
    };

    milestones.forEach(function (point) {
      var cx = xScale(point.x);
      var cy = yScale(point.tps);
      var offset = milestoneLabelOffsets[point.id];
      var labelX = cx + offset[0];
      var labelY = cy + offset[1];
      var titleText = milestoneTitles[point.id];
      var valueText = formatMilestoneTps(point) + ' TOKENS/S';
      var hitWidth = Math.max(titleText.length * 5.25, valueText.length * 5.6) + 12;
      var hitHeight = 32;
      var hitX = offset[0] < 0 ? labelX - hitWidth : labelX;
      var hitY = offset[1] < 0 ? labelY - 27 : labelY - 13;
      var labelGroup = make('g', {
        class: 'chart-milestone-target chart-milestone-animated',
        'data-point-id': point.id,
        tabindex: '0',
        role: 'button',
        'aria-label': 'Show ' + point.label + ' in the experiment log'
      });
      labelGroup.style.animationDelay = (1.35 + Math.sqrt(point.x / 366) * 1.7) + 's';
      labelGroup.appendChild(make('rect', {
        x: hitX,
        y: hitY,
        width: hitWidth,
        height: hitHeight,
        class: 'chart-milestone-hit'
      }));
      labelGroup.appendChild(make('line', {
        x1: cx + (offset[0] < 0 ? -5 : 5),
        y1: cy + (offset[1] < 0 ? -5 : 5),
        x2: offset[0] < 0 ? hitX + hitWidth : hitX,
        y2: offset[1] < 0 ? hitY + hitHeight : hitY,
        class: 'chart-milestone-leader'
      }));
      labelGroup.appendChild(make('text', {
        x: hitX + 6,
        y: hitY + 12,
        class: 'chart-milestone-title'
      }, titleText));
      labelGroup.appendChild(make('text', {
        x: hitX + 6,
        y: hitY + 25,
        class: 'chart-milestone-value'
      }, valueText));
      labelGroup.addEventListener('click', function () { selectManually(point); });
      labelGroup.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          selectManually(point);
        }
      });
      svg.appendChild(labelGroup);
    });

    if (activePoint) setActivePoint(activePoint, false);
  }

  function renderRail() {
    railList.replaceChildren();
    data.forEach(function (point) {
      var card = document.createElement('button');
      var meta = document.createElement('span');
      var title = document.createElement('strong');
      var detail = document.createElement('small');

      card.type = 'button';
      card.className = 'campaign-rail-card campaign-rail-card-' + point.status;
      card.dataset.pointId = point.id;
      card.style.animationDelay = (0.9 + Math.sqrt(point.x / 366) * 1.7) + 's';
      meta.textContent = (point.id === 'release-confirmation' ? 'CONFIRMATION' : 'ATTEMPT ' + String(point.x).padStart(3, '0')) +
        ' / ' + formatMilestoneTps(point) + ' TOKENS/S';
      title.textContent = point.label;
      detail.textContent = pointDetail(point);
      card.append(meta, title, detail);
      card.addEventListener('click', function () { selectManually(point); });
      railList.appendChild(card);
    });
    setActivePoint(milestones[0], false);
  }

  renderRail();
  draw();
  if ('ResizeObserver' in window) new ResizeObserver(draw).observe(root);
  else window.addEventListener('resize', draw);
})();

(function () {
  var root = document.querySelector('[data-research-system]');
  if (!root) return;

  var scrollRoot = root.closest('[data-rs-scroll]');
  var track = root.querySelector('[data-rs-track]');
  var svg = root.querySelector('.rs-track-lines');
  var path = root.querySelector('[data-rs-track-path]');
  var glowPath = root.querySelector('[data-rs-track-glow]');
  var invalidPath = root.querySelector('[data-rs-invalid-path]');
  var invalidLabel = root.querySelector('[data-rs-invalid-label]');
  var pulse = root.querySelector('[data-rs-pulse]');
  var nodes = Array.from(root.querySelectorAll('[data-rs-node]'));
  var meta = root.querySelector('[data-rs-meta]');
  var title = root.querySelector('[data-rs-title]');
  var copy = root.querySelector('[data-rs-copy]');
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var currentIndex = 0;
  var currentDistance = 0;
  var stageDistances = [];
  var pathLength = 0;
  var motionFrame = null;
  var activeScrollStop = -1;

  var details = {
    profile: ['AUTONOMOUS ATTEMPT / STEP 01', 'Read the global profile', 'Start from measured end-to-end cost and choose one bottleneck large enough to matter.'],
    hypothesis: ['AUTONOMOUS ATTEMPT / STEP 02', 'Name one mechanism', 'Explain the bottleneck and predict a measurable effect before changing the implementation.'],
    patch: ['AUTONOMOUS ATTEMPT / STEP 03', 'Build one candidate', 'Change one primary scheduling, layout, fusion, or dataflow idea in Python or CUDA.'],
    exactness: ['PROTECTED GATE / STEP 04', 'Try to falsify it', 'Check the forward pass and random-upstream gradients against an independent PyTorch oracle.'],
    benchmark: ['PROTECTED GATE / STEP 05', 'Measure the real workload', 'Only exact candidates reach the matched, production-shaped six-layer trainer benchmark.'],
    decision: ['AUTONOMOUS ATTEMPT / STEP 06', 'Keep, revert, and record', 'A material exact win becomes the baseline. Every other outcome still enters the experiment ledger.']
  };

  function pointForElement(element) {
    var elementRect = element.getBoundingClientRect();
    var trackRect = track.getBoundingClientRect();
    return {
      x: elementRect.left - trackRect.left + elementRect.width / 2,
      y: elementRect.top - trackRect.top + elementRect.height / 2
    };
  }

  function positionCircle(circle, selectedPath, distance) {
    var point = selectedPath.getPointAtLength(distance);
    circle.setAttribute('cx', point.x);
    circle.setAttribute('cy', point.y);
  }

  function drawTrack() {
    var width = track.clientWidth;
    var height = track.clientHeight;
    var centers = nodes.map(function (node) { return pointForElement(node.querySelector('span')); });
    var centerX = width / 2;
    var centerY = height / 2;
    var radius = Math.min(width, height) * .36;
    var startAngle = Math.PI * 7 / 6;
    var oppositeAngle = Math.PI / 6;
    var startX = centerX + Math.cos(startAngle) * radius;
    var startY = centerY + Math.sin(startAngle) * radius;
    var oppositeX = centerX + Math.cos(oppositeAngle) * radius;
    var oppositeY = centerY + Math.sin(oppositeAngle) * radius;
    var route = 'M' + startX + ' ' + startY +
      ' A' + radius + ' ' + radius + ' 0 1 1 ' + oppositeX + ' ' + oppositeY +
      ' A' + radius + ' ' + radius + ' 0 1 1 ' + startX + ' ' + startY + ' Z';
    svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    path.setAttribute('d', route);
    glowPath.setAttribute('d', route);
    pathLength = path.getTotalLength();
    stageDistances = nodes.map(function (_, index) { return pathLength * index / nodes.length; });

    var verify = centers[3];
    var invalidEndX = Math.min(width - 8, verify.x + 64);
    var invalidEndY = Math.max(18, verify.y - 40);
    var invalidControlX = (verify.x + invalidEndX) / 2;
    invalidPath.setAttribute('d', 'M' + verify.x + ' ' + verify.y +
      ' Q' + invalidControlX + ' ' + (invalidEndY - 12) + ' ' + invalidEndX + ' ' + invalidEndY);
    invalidLabel.setAttribute('x', invalidEndX);
    invalidLabel.setAttribute('y', invalidEndY - 6);
    invalidLabel.setAttribute('text-anchor', 'end');
    invalidLabel.textContent = 'FAIL → INVALID';

    if (!motionFrame) {
      currentDistance = stageDistances[currentIndex];
      positionCircle(pulse, path, currentDistance);
    }
  }

  function writeDetail(key) {
    var detail = details[key];
    meta.textContent = detail[0];
    title.textContent = detail[1];
    copy.textContent = detail[2];
  }

  function cancelMotion() {
    if (motionFrame) window.cancelAnimationFrame(motionFrame);
    motionFrame = null;
    pulse.classList.add('is-visible');
  }

  function paintStage(index, completedThrough) {
    currentIndex = index;
    nodes.forEach(function (node, nodeIndex) {
      var active = nodeIndex === index;
      node.classList.toggle('is-active', active);
      node.classList.toggle('is-complete', nodeIndex < completedThrough);
      node.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    writeDetail(nodes[index].dataset.rsNode);
  }

  function animatePulse(targetDistance) {
    cancelMotion();
    if (reduceMotion || !pathLength) {
      currentDistance = targetDistance;
      positionCircle(pulse, path, currentDistance);
      return;
    }

    var startDistance = currentDistance;
    var startedAt = null;
    function frame(time) {
      if (startedAt === null) startedAt = time;
      var progress = Math.min(1, (time - startedAt) / 320);
      var eased = 1 - Math.pow(1 - progress, 3);
      currentDistance = startDistance + (targetDistance - startDistance) * eased;
      positionCircle(pulse, path, currentDistance);
      if (progress < 1) motionFrame = window.requestAnimationFrame(frame);
      else motionFrame = null;
    }
    motionFrame = window.requestAnimationFrame(frame);
  }

  function showScrollStop(stop) {
    if (stop === activeScrollStop || !pathLength) return;
    activeScrollStop = stop;
    paintStage(stop, stop);
    animatePulse(stageDistances[stop]);
  }

  function stickyTop() {
    return parseFloat(window.getComputedStyle(root).top) || 0;
  }

  function scrollMetrics() {
    var rect = scrollRoot.getBoundingClientRect();
    var travel = Math.max(1, scrollRoot.offsetHeight - root.offsetHeight);
    return {
      progress: Math.max(0, Math.min(1, (stickyTop() - rect.top) / travel)),
      travel: travel,
      absoluteTop: window.scrollY + rect.top
    };
  }

  function syncToScroll() {
    var progress = scrollMetrics().progress;
    var stop = Math.min(nodes.length - 1, Math.floor(progress * nodes.length));
    showScrollStop(stop);
  }

  function scrollToStage(index, moveFocus) {
    var metrics = scrollMetrics();
    var progress = (index + .35) / nodes.length;
    var destination = metrics.absoluteTop - stickyTop() + metrics.travel * progress;
    window.scrollTo({ top: destination, behavior: reduceMotion ? 'auto' : 'smooth' });
    if (moveFocus) nodes[index].focus();
  }

  nodes.forEach(function (node, index) {
    node.addEventListener('click', function () { scrollToStage(index, false); });
    node.addEventListener('keydown', function (event) {
      if (event.key !== 'ArrowRight' && event.key !== 'ArrowDown' &&
          event.key !== 'ArrowLeft' && event.key !== 'ArrowUp') return;
      event.preventDefault();
      var direction = event.key === 'ArrowRight' || event.key === 'ArrowDown' ? 1 : -1;
      var target = Math.max(0, Math.min(nodes.length - 1, index + direction));
      scrollToStage(target, true);
    });
  });

  paintStage(0, 0);
  pulse.classList.add('is-visible');
  drawTrack();
  syncToScroll();

  if ('ResizeObserver' in window) {
    new ResizeObserver(function () {
      drawTrack();
      activeScrollStop = -1;
      syncToScroll();
    }).observe(track);
  } else {
    window.addEventListener('resize', function () {
      drawTrack();
      activeScrollStop = -1;
      syncToScroll();
    });
  }

  window.addEventListener('scroll', syncToScroll, { passive: true });

  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      if (!entries[0].isIntersecting) return;
      root.classList.add('is-visible');
      observer.disconnect();
    }, { threshold: .35 });
    observer.observe(root);
  } else {
    root.classList.add('is-visible');
  }
})();

(function () {
  var root = document.querySelector('[data-kda-explainer]');
  if (!root) return;

  var grid = root.querySelector('.kda-memory-grid');
  var controls = Array.from(root.querySelectorAll('[data-kda-step]'));
  var key = root.querySelector('[data-kda-key]');
  var value = root.querySelector('[data-kda-value]');
  var memoryState = root.querySelector('[data-kda-memory-state]');
  var action = root.querySelector('[data-kda-action]');
  var operation = root.querySelector('[data-kda-operation]');
  var inputNote = root.querySelector('[data-kda-input-note]');
  var outputNote = root.querySelector('[data-kda-output-note]');
  var caption = root.querySelector('[data-kda-caption]');

  var steps = [
    {
      key: 'ALPHA', value: '4', state: 'WRITE', action: 'ADD ASSOCIATION',
      input: 'Turn the token into an address and some content.',
      operation: '<span>address</span><strong>ALPHA</strong><span>now points toward</span><strong>4</strong>',
      output: 'The state grows no larger when the sequence does.',
      caption: 'Unlike standard attention, KDA does not retain every earlier key and value. It continually updates one compact working memory.',
      levels: [.08,.14,.05,.10,.05,.03, .12,.76,.42,.16,.05,.07, .05,.36,.88,.32,.09,.04, .03,.10,.26,.13,.05,.03]
    },
    {
      key: 'ALPHA′', value: '7', state: 'COLLISION', action: 'READ CURRENT ESTIMATE',
      input: 'A similar address arrives with different content.',
      operation: '<span>similar addresses overlap</span><strong>OLD + NEW</strong><span>so the memory first returns</span><strong>A MIXED ESTIMATE</strong>',
      output: 'A fixed state inevitably creates interference between associations.',
      caption: 'A finite memory cannot give every token its own private slot. Similar keys overlap, so blindly adding another value would preserve the error.',
      levels: [.10,.20,.09,.14,.06,.04, .18,.92,.70,.34,.12,.08, .08,.62,1,.58,.19,.06, .04,.19,.45,.30,.10,.04]
    },
    {
      key: 'TARGET', value: '7', state: 'CORRECT', action: 'ERASE ERROR · WRITE DIFFERENCE',
      input: 'The target is compared with what memory already predicts.',
      operation: '<span>update only the difference</span><strong>TARGET − ESTIMATE</strong><span>rather than adding</span><strong>ANOTHER FULL COPY</strong>',
      output: 'This prediction error is the “delta” in the delta rule.',
      caption: 'KDA reads its current prediction, measures the error, and writes the correction. The memory learns while the model reads.',
      levels: [.06,.12,.05,.11,.04,.03, .10,.58,.34,.28,.09,.06, .05,.27,.82,.68,.22,.07, .03,.08,.22,.48,.14,.04]
    },
    {
      key: 'ALPHA′', value: '7', state: 'FORGET', action: 'DECAY EACH CHANNEL',
      input: 'Before the next correction, the old state is selectively decayed.',
      operation: '<span>each memory channel gets</span><strong>ITS OWN RETENTION RATE</strong><span>preserving useful structure while</span><strong>FADING STALE CONTENT</strong>',
      output: 'KDA makes forgetting fine-grained instead of using one rate for the whole head.',
      caption: 'Some channels remember; others reset quickly. That fine-grained control is KDA’s main extension to Gated DeltaNet.',
      levels: [.02,.04,.02,.04,.02,.01, .09,.52,.31,.25,.08,.05, .02,.12,.36,.30,.10,.03, .03,.08,.22,.48,.14,.04]
    }
  ];

  var cells = steps[0].levels.map(function (_, index) {
    var cell = document.createElement('span');
    cell.className = 'kda-memory-cell';
    cell.dataset.cell = index;
    grid.appendChild(cell);
    return cell;
  });

  function render(index, focusControl) {
    var step = steps[index];
    key.textContent = step.key;
    value.textContent = step.value;
    memoryState.textContent = step.state;
    action.textContent = step.action;
    operation.innerHTML = step.operation;
    inputNote.textContent = step.input;
    outputNote.textContent = step.output;
    caption.textContent = step.caption;

    cells.forEach(function (cell, cellIndex) {
      var previous = Number(cell.style.getPropertyValue('--level')) || 0;
      var next = step.levels[cellIndex];
      cell.style.setProperty('--level', next);
      cell.style.setProperty('--cell-opacity', (.12 + next * .88).toFixed(3));
      cell.style.setProperty('--cell-glow', (next * .9).toFixed(2) + 'rem');
      cell.classList.toggle('is-updating', Math.abs(previous - next) > .22);
    });

    window.setTimeout(function () {
      cells.forEach(function (cell) { cell.classList.remove('is-updating'); });
    }, 540);

    controls.forEach(function (control, controlIndex) {
      control.setAttribute('aria-pressed', controlIndex === index ? 'true' : 'false');
    });
    if (focusControl) controls[index].focus();
  }

  controls.forEach(function (control, index) {
    control.addEventListener('click', function () { render(index, false); });
    control.addEventListener('keydown', function (event) {
      var nextIndex = index;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') nextIndex = (index + 1) % controls.length;
      else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') nextIndex = (index + controls.length - 1) % controls.length;
      else return;
      event.preventDefault();
      render(nextIndex, true);
    });
  });

  render(0, false);

  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      if (!entries[0].isIntersecting) return;
      root.classList.add('is-visible');
      observer.disconnect();
    }, { threshold: .12 });
    observer.observe(root);
  } else {
    root.classList.add('is-visible');
  }
})();
