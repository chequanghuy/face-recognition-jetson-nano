void FaceNetClassifier::getEmbedding(cv::Mat image, std::vector<struct Bbox> outputBbox,
        const string className) {
    
    //cv::resize(image, image, cv::Size(1280, 720), 0, 0, cv::INTER_CUBIC);
    // getCroppedFacesAndAlign(image, outputBbox);
    // if(!m_croppedFaces.empty()) {
    struct CroppedFace currFace;
    cv::resize(image, currFace.faceMat, cv::Size(160, 160), 0, 0, cv::INTER_CUBIC);
    // currFace.x1 = it->x1;
    // currFace.y1 = it->y1;
    // currFace.x2 = it->x2;
    // currFace.y2 = it->y2;            
    m_croppedFaces.push_back(currFace);
    preprocessFaces();
    doInference((float*)m_croppedFaces[0].faceMat.ptr<float>(0), m_output);
    // struct KnownID person;
    // person.className = className;
    // person.classNumber = m_classCount;
    // person.embeddedFace.insert(person.embeddedFace.begin(), m_output, m_output+128);

    // m_knownFaces.push_back(person);
    // m_classCount++;
    }
    // m_croppedFaces.clear();
}