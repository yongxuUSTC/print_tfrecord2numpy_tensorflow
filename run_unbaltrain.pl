use warnings;

# eavl has 4062 utterances
#for $i (0:10:4061)
$st=0;
while($st<5000)
{
$ed=$st+10;
system("python tfrecorder2cpickle_unbaltrain.py $st $ed");
$st=$st+10;
#die;
#if ($i>20) {die;}
  }
