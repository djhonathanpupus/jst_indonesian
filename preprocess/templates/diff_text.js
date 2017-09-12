var Diff = require('text-diff');
var diff = new Diff();
var textDiff = diff.main('texta', 'textb');
diff.prettyHtml(textDiff);