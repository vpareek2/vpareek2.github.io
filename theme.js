(function () {
  var root = document.documentElement;
  var key = 'theme';
  var btn = document.getElementById('theme-toggle');

  function apply(theme) {
    root.setAttribute('data-theme', theme);
  }

  function current() {
    return root.getAttribute('data-theme') || 'light';
  }

  function label() {
    if (!btn) return;
    btn.textContent = current() === 'dark' ? '\u2600\uFE0E' : '\u263E';
  }

  apply(localStorage.getItem(key) || 'light');
  label();

  if (btn) {
    btn.addEventListener('click', function () {
      var next = current() === 'dark' ? 'light' : 'dark';
      apply(next);
      localStorage.setItem(key, next);
      label();
    });
  }
})();
