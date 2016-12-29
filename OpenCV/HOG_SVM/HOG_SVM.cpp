#include "HOG_SVM.h"
#include "NBA_Methods.h"

int cnt = 1;

HOG_SVM::HOG_SVM(void)
{
	hog = HOGDescriptor();
	svm = CvSVM();
}

HOG_SVM::~HOG_SVM(void)
{
}

struct NBAInvoker
{
    NBAInvoker( const SVM* _svm, const Mat& _hog_res,		// const HOGDescriptor* _hog replaced by const HOG_SVM* _svm_hog
                double _hitThreshold, Size _windowSize, Size _winStride, Size _padding, Size _pimageSize,
                ConcurrentRectVector* _vec, ConcurrentDoubleVector* _weights=0) 
    {
        svm = (_svm);
		tmp = _hog_res;
        hitThreshold = _hitThreshold;
		winSize =  _windowSize ;
        winStride = _winStride;
        padding = _padding;
		pimageSize = _pimageSize ;
        hits = _vec;
        weights = _weights;
	}

    void operator()( const BlockedRange& range ) const
    {
        int idx, i1 = range.begin(), i2 = range.end();
//        double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1+1] : std::max(img.cols, img.rows);
//        Size maxSz(cvCeil(img.cols/minScale), cvCeil(img.rows/minScale));
//        Mat smallerImgBuf(maxSz, img.type());
//        vector<Point> locations;
//        vector<double> hitsWeights;

        for( idx = i1; idx < i2; idx++ )
        {
			double result = svm->predict(tmp.row(idx), true);
			if ( result >=  hitThreshold) 
			{
				int nwindowsX = (pimageSize.width - winSize.width)/winStride.width + 1;
				int y = idx / nwindowsX;
				int x = idx - nwindowsX*y;
				// return Rect( x*winStride.width, y*winStride.height, winSize.width, winSize.height );
				//hits.push_back(Point(x*winStride.width, y*winStride.height)- Point(padding))  ; // - Point(padding); // only for padding = Size(0,0)
				hits->push_back(Rect(Point(x*winStride.width, y*winStride.height)- Point(padding), winSize))  ; // - Point(padding); // only for padding = Size(0,0)
				weights->push_back(result) ; 
			// NOTE: NO NEED TO DO THIS: 0+ padding/winStride <= x,y <= max - padding/winStride : max-x = (pimageSize.width - winSize.width)/winStride.width + 1)  , ...
			//			Other words: 0 <= Point(x*winStride.width, y*winStride.height)- Point(padding) <= imageSize - window and 
			}
		}
	}
	const SVM* svm ;
    Mat tmp;
    double hitThreshold;
	Size winSize;
    Size winStride;
	Size padding;
	Size pimageSize ;
    ConcurrentRectVector* hits;
    ConcurrentDoubleVector* weights;
};


void HOG_SVM::detect(const Mat& img,
	vector<Point>& hits, vector<double>& weights, double hitThreshold, 
	Size winStride, Size padding, const vector<Point>& locations) const
{
hits.clear();
weights.clear();
vector<float> descriptors ;
hog.compute(img, descriptors, winStride, padding, locations) ;

Size pimageSize = img.size() + padding+padding; // size of the padded image
Size winSize = hog.winSize ;

int nWindows = ((pimageSize.width - winSize.width)/winStride.width + 1) *
				((pimageSize.height - winSize.height)/winStride.height + 1);

Mat tmp = Mat(descriptors).reshape(1, nWindows) ;

#if 0			// slow BUT Both Weights & hitThreshold

#if 0 // revised to a parrallel for
for (int idx=0; idx < nWindows ; ++idx)		// FIX: USE cv::parallel_for
{
	double result = svm.predict(tmp.row(idx), true);
	if ( result >=  hitThreshold) {
		int nwindowsX = (pimageSize.width - winSize.width)/winStride.width + 1;
		int y = idx / nwindowsX;
		int x = idx - nwindowsX*y;
		// return Rect( x*winStride.width, y*winStride.height, winSize.width, winSize.height );
		hits.push_back(Point(x*winStride.width, y*winStride.height)- Point(padding))  ; // - Point(padding); // only for padding = Size(0,0)
		weights.push_back(result) ; 
		// NOTE: NO NEED TO DO THIS: 0+ padding/winStride <= x,y <= max - padding/winStride : max-x = (pimageSize.width - winSize.width)/winStride.width + 1)  , ...
		//			Other words: 0 <= Point(x*winStride.width, y*winStride.height)- Point(padding) <= imageSize - window and 
	}
}
#else

	ConcurrentRectVector allCandidates;
    ConcurrentDoubleVector tempWeights;

	parallel_for(BlockedRange(0, nWindows), NBAInvoker(&(this->svm), tmp, hitThreshold, winSize, winStride, padding,pimageSize, &allCandidates, &tempWeights)); // this replaced by hog
	
	hits.clear() ;
    // std::copy(allCandidates.begin(), allCandidates.end(), back_inserter(hits));
	for (int i = 0; i < allCandidates.size() ; i++)
	{
		hits.push_back(allCandidates[i].tl()) ;
	}
    weights.clear();
    std::copy(tempWeights.begin(), tempWeights.end(), back_inserter(weights));


# endif

#else // Faster BUT NO Weights & NO hitThreshold	// FIX: should change line#2114 in svm.cpp :  float r = pointer->predict(&sample, true);

CvMat temp = tmp;

CvMat* results = cvCreateMat(nWindows,1, CV_32FC1) ;
svm.predict(&temp, results) ;  //  note double result = svm.predict(&temp, results) => result == results[0]
vector<double> result = vector<double>( results->data.fl, results->data.fl + nWindows ) ;

for (int idx=0; idx < nWindows ; ++idx)
{
	if ( result[idx] >=  hitThreshold) 
	{
		int nwindowsX = (pimageSize.width - winSize.width)/winStride.width + 1;
		int y = idx / nwindowsX;
		int x = idx - nwindowsX*y;
		// return Rect( x*winStride.width, y*winStride.height, winSize.width, winSize.height );
//		hits.push_back(Point(x*winStride.width, y*winStride.height)- Point(padding))  ; // - Point(padding); // only for padding = Size(0,0)
		weights.push_back(result[idx]) ; 
		// NOTE: NO NEED TO DO THIS: 0+ padding/winStride <= x,y <= max - padding/winStride : max-x = (pimageSize.width - winSize.width)/winStride.width + 1)  , ...
		//			Other words: 0 <= Point(x*winStride.width, y*winStride.height)- Point(padding) <= imageSize - window and 

		// OLD CODE
//		if ( weights[idx] ==  1) 
//		{
//		int nwindowsX = (imageSize.width - winSize.width)/winStride.width + 1;
//		int y = idx / nwindowsX;
//		int x = idx - nwindowsX*y;
		// return Rect( x*winStride.width, y*winStride.height, winSize.width, winSize.height );
		hits.push_back(Point(x*winStride.width, y*winStride.height) - Point(padding)) ; // only for padding = Size(0,0)
	}
}

cvReleaseMat(&results) ;


#endif

return ;
}

#if 1

void HOG_SVM::detect(const Mat& img, vector<Point>& hits, double hitThreshold, 
						Size winStride, Size padding, const vector<Point>& locations) const
{
	vector<double> weightsV;
	detect(img, hits, weightsV, hitThreshold, winStride, padding, locations);
}


struct HOGInvoker
{
    HOGInvoker( const HOG_SVM* _svm_hog, const Mat& _img,		// const HOGDescriptor* _hog replaced by const HOG_SVM* _svm_hog
                double _hitThreshold, Size _winStride, Size _padding,
                const double* _levelScale, ConcurrentRectVector* _vec, 
                ConcurrentDoubleVector* _weights=0, ConcurrentDoubleVector* _scales=0 ) 
    {
        hogI = &(_svm_hog->hog);
		svm_hog = _svm_hog ;
        img = _img;
        hitThreshold = _hitThreshold;
        winStride = _winStride;
        padding = _padding;
        levelScale = _levelScale;
        vec = _vec;
        weights = _weights;
        scales = _scales;
    }

    void operator()( const BlockedRange& range ) const
    {
        int i, i1 = range.begin(), i2 = range.end();
        double minScale = i1 > 0 ? levelScale[i1] : i2 > 1 ? levelScale[i1+1] : std::max(img.cols, img.rows);
        Size maxSz(cvCeil(img.cols/minScale), cvCeil(img.rows/minScale));
        Mat smallerImgBuf(maxSz, img.type());
        vector<Point> locations;
        vector<double> hitsWeights;

        for( i = i1; i < i2; i++ )
        {
            double scale = levelScale[i];
            Size sz(cvRound(img.cols/scale), cvRound(img.rows/scale));
            Mat smallerImg(sz, img.type(), smallerImgBuf.data);
            if( sz == img.size() )
                smallerImg = Mat(sz, img.type(), img.data, img.step);
            else
                resize(img, smallerImg, sz);
			svm_hog->detect(smallerImg, locations, hitsWeights, hitThreshold, winStride, padding);
			// SHOW_RECT(smallerImg, locations, "BeTest NBA_HOG_SVM");
			// waitKey() ;
            Size scaledWinSize = Size(cvRound(svm_hog->hog.winSize.width*scale), cvRound(svm_hog->hog.winSize.height*scale));
            for( size_t j = 0; j < locations.size(); j++ )
            {
                vec->push_back(Rect(cvRound(locations[j].x*scale),
                                    cvRound(locations[j].y*scale),
                                    scaledWinSize.width, scaledWinSize.height));
                if (scales) {
                    scales->push_back(scale);
                }
            }
            
            if (weights && (!hitsWeights.empty()))
            {
                for (size_t j = 0; j < locations.size(); j++)
                {
                    weights->push_back(hitsWeights[j]);
                }
            }        
        }
    }

    const HOGDescriptor* hogI;
	const HOG_SVM* svm_hog ;
    Mat img;
    double hitThreshold;
    Size winStride;
    Size padding;
    const double* levelScale;
    ConcurrentRectVector* vec;
    ConcurrentDoubleVector* weights;
    ConcurrentDoubleVector* scales;
};


void HOG_SVM::detectMultiScale(
    const Mat& img, vector<Rect>& foundLocations, vector<double>& foundWeights,
    double hitThreshold, Size winStride, Size padding,
    double scale0, double finalThreshold, bool useMeanshiftGrouping) const  
{
    double scale = 1.;
    int levels = 0;
	int nlevels = hog.nlevels ; // NBA or The class should be derived from HOGDescriptor

    vector<double> levelScale;
    for( levels = 0; levels < nlevels; levels++ )		// This block is for producing the scales (maximum would be nlevels which is an initial parameter while initializing the hog object )
    {
        levelScale.push_back(scale);
        if( cvRound(img.cols/scale) < hog.winSize.width ||
            cvRound(img.rows/scale) < hog.winSize.height ||
            scale0 <= 1 )
            break;
        scale *= scale0;
    }
    levels = std::max(levels, 1);
    levelScale.resize(levels);

    ConcurrentRectVector allCandidates;
    ConcurrentDoubleVector tempScales;
    ConcurrentDoubleVector tempWeights;
    vector<double> foundScales;
    
    parallel_for(BlockedRange(0, (int)levelScale.size()),
                 HOGInvoker(this, img, hitThreshold, winStride, padding, &levelScale[0], &allCandidates, &tempWeights, &tempScales)); // this replaced by hog

    std::copy(tempScales.begin(), tempScales.end(), back_inserter(foundScales));
    foundLocations.clear();
    std::copy(allCandidates.begin(), allCandidates.end(), back_inserter(foundLocations));
    foundWeights.clear();
    std::copy(tempWeights.begin(), tempWeights.end(), back_inserter(foundWeights));

	// NBA Test
	SHOW_RECT(img, foundLocations, "Before_NBA_HOG_SVM",cnt, false);
	cnt++ ;
	//waitKey() ;
    if ( useMeanshiftGrouping )
    {
        groupRectangles_meanshift(foundLocations, foundWeights, foundScales, finalThreshold, hog.winSize);
    }
    else
    {
        vector<int> weights2 ;
		groupRectangles(foundLocations, weights2, (int)finalThreshold, 0.2);
		//groupRectangles(foundLocations,(int)finalThreshold, 0.2);
    }
}

void HOG_SVM::detectMultiScale(const Mat& img, vector<Rect>& foundLocations, 
                                     double hitThreshold, Size winStride, Size padding,
                                     double scale0, double finalThreshold, bool useMeanshiftGrouping) const  
{
    vector<double> foundWeights;
    detectMultiScale(img, foundLocations, foundWeights, hitThreshold, winStride, 
                     padding, scale0, finalThreshold, useMeanshiftGrouping);
}

#endif