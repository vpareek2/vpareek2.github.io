(function () {
  var alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%&*+-=?<>[]{}';
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var states = [];
  var contentFrame = 0;
  var contentStarted = performance.now();
  var themeToggle = document.querySelector('[data-theme-toggle]');

  function isLightTheme() {
    return document.documentElement.dataset.theme === 'light';
  }

  function updateThemeToggle() {
    if (!themeToggle) return;
    var light = isLightTheme();
    var label = light ? 'Switch to dark mode' : 'Switch to light mode';
    themeToggle.setAttribute('aria-label', label);
    themeToggle.setAttribute('aria-pressed', String(light));
    themeToggle.setAttribute('title', label);
  }

  updateThemeToggle();

  if (themeToggle) {
    themeToggle.addEventListener('click', function () {
      var nextTheme = isLightTheme() ? 'dark' : 'light';
      if (nextTheme === 'light') document.documentElement.dataset.theme = 'light';
      else delete document.documentElement.dataset.theme;
      try { localStorage.setItem('veer-theme', nextTheme); } catch (error) {}
      updateThemeToggle();
      if (ctx) drawField(false);
    });
  }

  function randomCharacter() {
    return alphabet.charAt(Math.floor(Math.random() * alphabet.length));
  }

  function buildElement(element) {
    var original = element.textContent;
    var delay = Number(element.dataset.delay || 0);
    var state = { element: element, characters: [], complete: false };
    element.textContent = '';
    element.setAttribute('aria-label', original);

    Array.from(original).forEach(function (character, index) {
      if (/\s/.test(character)) {
        element.appendChild(document.createTextNode(character));
        return;
      }

      var span = document.createElement('span');
      span.className = 'matrix-char';
      span.textContent = randomCharacter();
      span.setAttribute('aria-hidden', 'true');
      element.appendChild(span);
      state.characters.push({
        node: span,
        final: character,
        settleAt: delay + 420 + Math.random() * 820 + index * 5,
        settled: false
      });
    });

    element.classList.add('matrix-built');
    states.push(state);

    if (element.hasAttribute('data-repeat')) {
      var repeat = function () {
        var now = performance.now() - contentStarted;
        state.complete = false;
        state.characters.forEach(function (item) {
          item.settled = false;
          item.node.classList.remove('settled');
          item.node.classList.remove('flipping');
          item.node.textContent = randomCharacter();
          item.settleAt = now + 90 + Math.random() * 420;
        });
        startContentLoop();
      };
      element.addEventListener('mouseenter', repeat);
      element.addEventListener('focus', repeat);
    }
  }

  function contentLoop(time) {
    var elapsed = time - contentStarted;
    var hasUnsettled = false;

    states.forEach(function (state) {
      if (state.complete) return;
      var stateComplete = true;

      state.characters.forEach(function (item) {
        if (item.settled) return;
        if (reduceMotion || elapsed >= item.settleAt) {
          item.node.textContent = item.final;
          item.node.classList.remove('flipping');
          item.node.classList.add('settled');
          item.settled = true;
        } else {
          stateComplete = false;
          hasUnsettled = true;
          if (item.settleAt - elapsed < 190) item.node.classList.add('flipping');
          else item.node.classList.remove('flipping');
          if (Math.random() < .42) item.node.textContent = randomCharacter();
        }
      });

      state.complete = stateComplete;
    });

    if (hasUnsettled) {
      window.setTimeout(function () {
        contentFrame = window.requestAnimationFrame(contentLoop);
      }, 40);
    } else {
      contentFrame = 0;
    }
  }

  function startContentLoop() {
    if (contentFrame) return;
    contentFrame = window.requestAnimationFrame(contentLoop);
  }

  Array.prototype.forEach.call(document.querySelectorAll('[data-matrix]'), buildElement);
  startContentLoop();

  var canvas = document.getElementById('character-field');
  if (!canvas) return;
  var ctx = canvas.getContext('2d');
  var cells = [];
  var width = 0;
  var height = 0;
  var dpr = 1;
  var cellWidth = 14;
  var cellHeight = 18;
  var columns = 0;
  var rows = 0;
  var fieldTimer = null;

  function buildField() {
    columns = Math.ceil(width / cellWidth) + 1;
    rows = Math.ceil(height / cellHeight) + 1;
    cells = [];
    for (var index = 0; index < columns * rows; index += 1) {
      cells.push({ character: randomCharacter(), tone: Math.random() });
    }
  }

  function drawField(change) {
    var styles = window.getComputedStyle(document.documentElement);
    var fieldRgb = styles.getPropertyValue('--field-rgb').trim() || '72, 111, 78';
    var alphaMin = Number(styles.getPropertyValue('--field-alpha-min')) || .045;
    var alphaRange = Number(styles.getPropertyValue('--field-alpha-range')) || .045;
    ctx.clearRect(0, 0, width, height);
    ctx.font = '400 10px ui-monospace, SFMono-Regular, Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    cells.forEach(function (cell, index) {
      if (change && Math.random() < .022) cell.character = randomCharacter();
      var column = index % columns;
      var row = Math.floor(index / columns);
      var alpha = alphaMin + cell.tone * alphaRange;
      ctx.fillStyle = 'rgba(' + fieldRgb + ',' + alpha + ')';
      ctx.fillText(cell.character, column * cellWidth + cellWidth * .5, row * cellHeight + cellHeight * .5);
    });
  }

  function resize() {
    var rect = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = Math.max(1, rect.width);
    height = Math.max(1, rect.height);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    buildField();
    drawField(false);
  }

  resize();
  window.addEventListener('resize', resize);

  if (!reduceMotion) {
    fieldTimer = window.setInterval(function () { drawField(true); }, 120);
  }

  document.addEventListener('visibilitychange', function () {
    if (document.hidden && fieldTimer !== null) {
      window.clearInterval(fieldTimer);
      fieldTimer = null;
    } else if (!document.hidden && !reduceMotion && fieldTimer === null) {
      fieldTimer = window.setInterval(function () { drawField(true); }, 120);
    }
  });
})();
