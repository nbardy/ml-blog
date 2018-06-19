// From stackoverflow 
// https://stackoverflow.com/questions/521295/seeding-the-random-number-generator-in-javascript
//
var mask = 0xffffffff;

// Takes any integer
export function seededRandom(i) {
  var m_w = i;
  var m_z = 987654321;


    // Returns number between 0 (inclusive) and 1.0 (exclusive),
    // just like Math.random().
  return () =>
  {
    m_z = (36969 * (m_z & 65535) + (m_z >> 16)) & mask;
    m_w = (18000 * (m_w & 65535) + (m_w >> 16)) & mask;
    var result = ((m_z << 16) + m_w) & mask;
    result /= 4294967296;
    return result + 0.5;
  };
}