#!/usr/bin/perl -w

use strict;
use warnings;
use Carp;
# use Cwd 'abs_path';
use Cwd;
use File::Basename;

my $workdir = dirname(__FILE__);
#"perl "+script_path+"/generate-other.pl "+db_tool_dir+" "+fasta+" "+outdir+" "+uniref90_dir+"/uniref90")
####################################################################################################
# my $db_tool_dir = shift;
my $fasta  = shift;
my $outdir = shift;
my $unirefdb = shift;

if (not $fasta or not -f $fasta){
	print "Fasta file $fasta does not exist!\n" if ($fasta and not -f $fasta);
	print "Usage: $0 <db_tools_dir> <fasta> <output-directory> <unirefdb>\n";
	exit(1);
}

if (not $outdir){
	print 'Output directory not defined!';
	print "Usage: $0 <db_tools_dir> <fasta> <output-directory> <unirefdb>\n";
	exit(1);
}

# if (not $db_tool_dir){
# 	print 'Databases directory not defined!';
# 	print "Usage: $0 <db_tools_dir> <fasta> <output-directory> <unirefdb>\n";
# 	exit(1);
# }

use constant{
	FORMATDBPATH      => '/blast-2.2.26/bin/formatdb', # added by Tianqi on 10/03/2017
	BLASTPATH    => '/blast-2.2.26/bin',
};

confess "Oops!! Blast Path not found at ".$workdir.BLASTPATH         if not -d $workdir.BLASTPATH;

print $workdir;
print "\n";
print "Input: $fasta\n";
print "L    : ".length(seq_fasta($fasta))."\n";
print "Seq  : ".seq_fasta($fasta)."\n\n";


my $id = basename($fasta, ".fasta");
system_cmd("mkdir -p $outdir") if not -d $outdir;
if(not -e $outdir."/".$id.".fasta"){
	print $outdir."/".$id.".fasta";
	system_cmd("cp $fasta $outdir/");
	system_cmd("echo \">$id\" > $outdir/$id.fasta");
	system_cmd("echo \"".seq_fasta($fasta)."\" >> $outdir/$id.fasta");
}

# $outdir = abs_path($outdir);
chdir $outdir or confess $!;
$fasta = basename($fasta);


my $seq_len = length(seq_fasta($fasta));
####################################################################################################
print "\n\n";
print "Generating PSSM..\n";
system_cmd("mkdir -p $outdir/pssm");
chdir $outdir."/pssm" or confess $!;
system_cmd("cp ../$fasta ./");
if ( -s "$id.pssm"){
	print "Looks like .pssm file is already here.. skipping..\n";
}
else{
	generate_pssm($fasta, "$id.pssm", "$id.psiblast.output");
	if( ! -e "$id.pssm") {
		print "Running less stringent version of PSSM generation..\n";
		generate_pssm_less_stringent($fasta, "$id.pssm", "$id.psiblast.output");
	}

  ### Modified by Tianqi, 10-03-2017, because some proteins couldn't find hits, so generete pseudo pssm here, need improve later
	if(! -e "$id.pssm"){
		print "Running pseudo version of PSSM generation..\n";
		generate_pssm_pseudo($fasta, "$id.pssm", "$id.psiblast.output");
	}
}
chdir $outdir or confess $!;

my $pssm_fname = $outdir.'/pssm/'.$id.'.pssm';
print($pssm_fname);
my @lines =();
open PSSM, "<" . $pssm_fname or die "Couldn't open pssm file $pssm_fname ".$!."\n";
@lines = <PSSM>;
chomp(@lines);
close PSSM;

my ($odds_ref, $inf_ref) = scaled_log_odds_n_inf(@lines);
my @odds = @{$odds_ref};
$pssm_fname = $outdir.'/'.$id.'_pssm.txt';
open PSSM, ">" . $pssm_fname or die "Couldn't open pssm file $pssm_fname ".$!."\n";
print PSSM "# PSSM\n";
for(my $l = 0; $l < @{$odds[0]}; $l++) {
	for(my $i = 0; $i < $seq_len; $i++) {
		#need to make sure that this is correct!!
		my $xx = $odds[$i][$l];
		$xx = 0 if not defined $xx;
		print PSSM " $xx";
	}
	print PSSM "\n";
}
close PSSM;

####################################################################################################
sub generate_pssm{
	my $FASTA = shift;
	my $PSSM = shift;
	my $REPORT = shift;
	print "Running PSI-Blast with $unirefdb...\n";
	#system_cmd($db_tool_dir.BLASTPATH. "/psiblast -query ".$FASTA." -evalue .001 -inclusion_ethresh .002 -db $unirefdb -num_iterations 3 -outfmt 0 -out $REPORT -seg yes -out_ascii_pssm $PSSM");
	system_cmd($workdir.BLASTPATH. "/blastpgp -a 8 -e 0.001 -j 3 -h 0.002 -d $unirefdb -i $FASTA -Q $PSSM >& $REPORT");
}

####################################################################################################
sub generate_pssm_less_stringent{
	my $FASTA = shift;
	my $PSSM = shift;
	my $REPORT = shift;
	print "Running PSI-Blast with $unirefdb...\n";
	#system_cmd($db_tool_dir.BLASTPATH. "/psiblast -query ".$FASTA." -evalue 10 -inclusion_ethresh 10 -db $unirefdb -num_iterations 3 -outfmt 0 -out $REPORT -num_alignments 2000 -out_ascii_pssm $PSSM");
	system_cmd($workdir.BLASTPATH. "/blastpgp -a 8 -e 10 -j 3 -h 10 -d $unirefdb -i $FASTA -Q $PSSM >& $REPORT");
}

####################################################################################################

sub generate_pssm_pseudo{
	my $FASTA = shift;
	my $PSSM = shift;
	my $REPORT = shift;
	print "Running PSI-Blast with pseudo version...\n";
	system_cmd($workdir.FORMATDBPATH. "  -i ".$FASTA);
	#system_cmd($workdir.BLASTPATH. "/psiblast -query ".$FASTA." -evalue .001 -inclusion_ethresh .002 -db $FASTA -num_iterations 3 -outfmt 0 -out $REPORT -seg yes -out_ascii_pssm $PSSM");
	system_cmd($workdir.BLASTPATH. "/blastpgp -a 8 -e 0.001 -j 3 -h 0.002 -d $FASTA -i $FASTA -Q $PSSM >& $REPORT");
}

####################################################################################################
sub seq_fasta{
	my $file_fasta = shift;
	confess "ERROR! Fasta file $file_fasta does not exist!" if not -f $file_fasta;
	my $seq = "";
	open FASTA, $file_fasta or confess $!;
	while (<FASTA>){
		next if (substr($_,0,1) eq ">");
		chomp $_;
		$_ =~ tr/\r//d; # chomp does not remove \r
		$seq .= $_;
	}
	close FASTA;
	return $seq;
}

####################################################################################################
sub system_cmd{
	my $command = shift;
	my $log = shift;
	confess "EXECUTE [$command]?\n" if (length($command) < 5  and $command =~ m/^rm/);
	if(defined $log){
		system("$command &> $log");
	}
	else{
		print "[[Executing: $command]]\n";
		system($command);
	}
	if($? != 0){
		my $exit_code  = $? >> 8;
		confess "ERROR!! Could not execute [$command]! \nError message: [$!]";
	}
}

#######################################################################
# Name : scaled_log_odds_n_inf 
# Takes: An array of lines coming from a pssm
# Returns: A 2d array for scaled versions of the log odds of each 
#  residue at each positions and an array of scaled information  
#  values at each position
######################################################################
sub scaled_log_odds_n_inf {
  my @lines = @_;
  my @pssm =();
  my @inf = ();

  # Get rid of header
  shift @lines; shift @lines; shift @lines;

  while(my $line_t = shift @lines) {
    if($line_t =~ m/^\s*$/) {
      last; 
    }
    my @fields = ();
    for(my $i = 0; $i < 20; $i++) {
       my $val = substr($line_t, 9+$i*3, 3);
       push(@fields, sprintf("%.4f", 1/(1+exp(-1*$val))));
    }
    push(@pssm, [@fields]);

    my $inf_t = substr($line_t, 151, 5);
    if($inf_t > 6) {
      push(@inf, 1.0);
    } elsif( $inf_t < 0) {
      push(@inf, 0);
    } else {
      push(@inf, sprintf("%.4f", ($inf_t/6.0)));
    }
  }

  return(\@pssm, \@inf);

}

