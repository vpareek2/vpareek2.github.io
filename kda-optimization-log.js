(function () {
  var root = document.querySelector('[data-optimization-log]');
  if (!root) return;

  var screen = root.querySelector('[data-log-screen]');
  var entries = Array.from(root.querySelectorAll('.optimization-log-entry'));
  var state = root.querySelector('[data-log-state]');
  var progress = root.querySelector('[data-log-progress]');
  var phase = root.querySelector('[data-log-phase]');
  var best = root.querySelector('[data-log-best]');
  var toggle = root.querySelector('[data-log-toggle]');
  var showAll = root.querySelector('[data-log-all]');
  var speedControl = root.querySelector('[data-log-speed]');
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var index = 0;
  var timer = null;
  var started = false;
  var playing = false;
  var completed = false;
  var userPaused = false;
  var offscreenPaused = false;
  var runningBest = 0;
  var playbackSpeed = Number(speedControl.value);
  var lastRevealed = null;

  function formatNumber(value) {
    return Math.round(value).toLocaleString('en-US');
  }

  function setControls() {
    if (completed) {
      toggle.textContent = 'COMPLETE';
      toggle.disabled = true;
      showAll.textContent = 'REPLAY';
      return;
    }
    toggle.disabled = false;
    toggle.textContent = playing ? 'PAUSE' : (started ? 'RESUME' : 'START');
    showAll.textContent = 'SHOW ALL';
  }

  function updateReadout(entry) {
    progress.textContent = 'TRACE ' + String(index + 1).padStart(2, '0') + ' / ' + entries.length;
    if (entry.dataset.logPhase) phase.textContent = entry.dataset.logPhase;
    if (entry.dataset.logBest) {
      runningBest = Math.max(runningBest, Number(entry.dataset.logBest));
      best.textContent = 'BEST ' + formatNumber(runningBest) + ' TOKENS/S';
    }
  }

  function follow(entry) {
    var target = entry.offsetTop + entry.offsetHeight - screen.clientHeight + 44;
    if (target <= screen.scrollTop) return;
    screen.scrollTo({ top: target, behavior: reduceMotion ? 'auto' : 'smooth' });
  }

  function delayFor(entry) {
    if (entry.classList.contains('log-milestone')) return 2400;
    if (entry.classList.contains('log-human') || entry.classList.contains('log-outside')) return 1350;
    if (entry.classList.contains('log-phase')) return 850;
    return 620;
  }

  function finish() {
    window.clearTimeout(timer);
    timer = null;
    playing = false;
    completed = true;
    state.textContent = 'COMPLETE';
    progress.textContent = 'TRACE ' + entries.length + ' / ' + entries.length;
    phase.textContent = 'COMPLETE';
    setControls();
  }

  function revealNext() {
    if (!playing) return;
    if (index >= entries.length) {
      finish();
      return;
    }

    var entry = entries[index];
    lastRevealed = entry;
    entry.classList.add('is-revealed');
    updateReadout(entry);
    index += 1;
    follow(entry);
    timer = window.setTimeout(revealNext, delayFor(entry) / playbackSpeed);
  }

  function play() {
    if (completed || playing) return;
    started = true;
    playing = true;
    userPaused = false;
    offscreenPaused = false;
    state.textContent = 'RUNNING';
    setControls();
    revealNext();
  }

  function pause(byUser) {
    if (!playing) return;
    window.clearTimeout(timer);
    timer = null;
    playing = false;
    if (byUser) userPaused = true;
    state.textContent = byUser ? 'PAUSED' : 'WAITING';
    setControls();
  }

  function revealEverything() {
    window.clearTimeout(timer);
    timer = null;
    entries.forEach(function (entry) { entry.classList.add('is-revealed'); });
    index = entries.length;
    runningBest = entries.reduce(function (value, entry) {
      return Math.max(value, Number(entry.dataset.logBest || 0));
    }, 0);
    best.textContent = 'BEST ' + formatNumber(runningBest) + ' TOKENS/S';
    screen.scrollTop = 0;
    finish();
  }

  function replay() {
    window.clearTimeout(timer);
    entries.forEach(function (entry) {
      entry.classList.remove('is-revealed', 'is-expanded');
      var button = entry.querySelector('button[aria-expanded]');
      if (button) button.setAttribute('aria-expanded', 'false');
    });
    index = 0;
    runningBest = 0;
    lastRevealed = null;
    completed = false;
    started = true;
    playing = false;
    userPaused = false;
    offscreenPaused = false;
    screen.scrollTop = 0;
    progress.textContent = 'TRACE 00 / ' + entries.length;
    phase.textContent = 'INITIALIZING';
    best.textContent = 'BEST —';
    state.textContent = 'READY';
    play();
  }

  root.classList.add('is-prepared');
  progress.textContent = 'TRACE 00 / ' + entries.length;
  setControls();

  root.querySelectorAll('.log-expandable button').forEach(function (button) {
    button.addEventListener('click', function () {
      pause(true);
      var entry = button.closest('.log-expandable');
      var expanded = !entry.classList.contains('is-expanded');
      entry.classList.toggle('is-expanded', expanded);
      button.setAttribute('aria-expanded', String(expanded));
    });
  });

  toggle.addEventListener('click', function () {
    if (playing) pause(true);
    else if (!completed) play();
  });

  showAll.addEventListener('click', function () {
    if (completed) replay();
    else revealEverything();
  });

  speedControl.addEventListener('change', function () {
    playbackSpeed = Number(speedControl.value) || 1;
    if (!playing || !lastRevealed) return;
    window.clearTimeout(timer);
    timer = window.setTimeout(revealNext, delayFor(lastRevealed) / playbackSpeed);
  });

  if (reduceMotion) {
    revealEverything();
    toggle.textContent = 'COMPLETE';
    showAll.textContent = 'SHOWING ALL';
    showAll.disabled = true;
    speedControl.disabled = true;
    return;
  }

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (records) {
      var visible = records[0].isIntersecting && records[0].intersectionRatio >= .35;
      if (visible) root.classList.add('is-visible');
      if (visible && !started) play();
      else if (visible && offscreenPaused && !userPaused && !completed) play();
      else if (!visible && playing) {
        offscreenPaused = true;
        pause(false);
      }
    }, { threshold: [0, .35, .7] }).observe(root);
  } else {
    root.classList.add('is-visible');
    play();
  }
})();
