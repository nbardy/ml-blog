export function newElement(type,attributes) {
  var ele = document.createElement(type)

  for(var k in attributes) {
    ele[k] = attributes[k];
  }

  return ele;

}
